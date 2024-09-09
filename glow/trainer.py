import os
import torch
import datetime
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
from .utils import savemodel, merge_leading_dims, sympot
from .models import Glow

cpu = torch.device('cpu')

class Trainer():
    def __init__(self, graph, dataset, device, vit_device, 
                 log_dir = None, 
                 vit_chiraln2 = None, 
                 vit_chiraln2_normc = None, 
                 vit_chiraln3 = None, 
                 vit_chiraln3_normc = None, 
                 lr = 1e-4, num_epoch = 1000, batch_size = 3, log_gap = 10
                 ):

        self.log_gap = log_gap

        # create log dir
        date = str(datetime.datetime.now())
        date = date[:date.rfind(':')].replace('-','')\
                .replace(':','').replace(' ','')
        self.log_dir = log_dir if log_dir is not None \
                else os.path.join('result', 'log_'+date)
        self.model_dir = os.path.join(self.log_dir, 'models')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.batch_size = batch_size
        self.data_loader = DataLoader(
                dataset, batch_size = self.batch_size, 
                shuffle = False, drop_last = False, 
                num_workers = 2*batch_size
                )
        self.num_epoch = num_epoch

        self.device = torch.device(device)
        self.graph = graph.to(self.device)
        self.optimizer = optim.Adam(self.graph.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr = lr, epochs = num_epoch, 
                steps_per_epoch = len(self.data_loader)
                )

        # load ViT models
        if (vit_chiraln2 is not None) and (vit_chiraln3 is not None):
            self.vit_device = torch.device(vit_device)
            self.vit_chiraln2 = vit_chiraln2.to(vit_device)
            self.vit_chiraln3 = vit_chiraln3.to(vit_device)
            self.vit_chiraln2_scale = vit_chiraln2_normc[0]
            self.vit_chiraln2_shift = vit_chiraln2_normc[1]
            self.vit_chiraln3_scale = vit_chiraln3_normc[0]
            self.vit_chiraln3_shift = vit_chiraln3_normc[1]
            self.vit = True

            # the lambdas used for ViT combined training 
            self.rand_lamblist = torch.arange(400., 601., 5., 
                                            device = self.device)
            self.fix_lamblist = torch.tensor([450., 500., 550.], 
                                            device = self.device)
        else:
            self.vit = False


        # create log file
        self.lossf = os.path.join(self.log_dir, 'loglike.dat')
        open(self.lossf, 'w').close()

    def train(self):
        self.graph.train()
        
        global_step = 0
        for epoch in tqdm(range(self.num_epoch), ncols = 52):
            for i_batch, batch in enumerate(tqdm(self.data_loader, ncols = 52)):
                loss = 0.0
                self.graph.zero_grad()
                self.optimizer.zero_grad()

                # move samples to graph's device
                for k in batch:
                    batch[k] = batch[k].to(self.device, dtype = torch.float32)
                # x: samples. y_onehot: labels (chiral order, lambda)
                x = batch['x']
                y_onehot = batch['y_onehot'].to(
                        self.device, dtype=torch.float32)

                B, C, H, W = x.shape

                # initialize actnorm at the begining
                if global_step == 0:
                    self.graph(x, y_onehot)

                # forward z = Glow(x). transform samples to latent space
                z, nll = self.graph(x = x, y_onehot = y_onehot)

                loglikelihoodloss = Glow.loss_generative(nll)
                loss += loglikelihoodloss.to(cpu)

                if self.vit:
                    # create the labels (2, lambds) for ViT regression training
                    rand_lamb_id = torch.randint(
                            0, len(self.rand_lamblist), (2,))
                    fix_lamb_id = torch.randint(
                            0, len(self.fix_lamblist), (1,))
                    lamblist = torch.cat(
                            (self.rand_lamblist[rand_lamb_id], 
                            self.fix_lamblist[fix_lamb_id])
                        )
                    lamblist = (lamblist - 400.)/(600. - 400.)

                    pred_label = torch.zeros(
                            2*len(lamblist), 2, device=self.device)
                    pred_label[:len(lamblist), 0] = 2.0
                    pred_label[len(lamblist):, 0] = 3.0
                    pred_label[:len(lamblist), 1] = lamblist
                    pred_label[len(lamblist):, 1] = lamblist

                    # generate chiral v=2 and v=3 potentials
                    pred_samples = self.graph.predsample(
                            y_pred = pred_label, num_samples = 1
                            )
                    pred_samples = merge_leading_dims(
                            x=pred_samples, num_dims=2)
                    pred_samples = sympot(pred_samples).to(self.vit_device)
                    pred_chiraln2 = pred_samples[:len(lamblist)]
                    pred_chiraln3 = pred_samples[len(lamblist):]

                    # use ViT model extract lambdas
                    self.vit_chiraln2.eval()
                    self.vit_chiraln3.eval()

                    vit_chiraln2_leclamb = self.vit_chiraln2(
                            pred_chiraln2[:, :, ::2, ::2])
                    vit_chiraln2_leclamb = (
                            vit_chiraln2_leclamb - self.vit_chiraln2_shift)/\
                                    self.vit_chiraln2_scale
                    vit_chiraln2_lamb = vit_chiraln2_leclamb[:, -1]\
                            .to(self.device)
                    vit_chiraln2_lamb = (vit_chiraln2_lamb-400.)/(600.-400.)

                    vit_chiraln3_leclamb = self.vit_chiraln3(
                            pred_chiraln3[:, :, ::2, ::2])
                    vit_chiraln3_leclamb = (
                            vit_chiraln3_leclamb - self.vit_chiraln3_shift)\
                                    /self.vit_chiraln3_scale
                    vit_chiraln3_lamb = vit_chiraln3_leclamb[:, -1]\
                            .to(self.device)
                    vit_chiraln3_lamb = (vit_chiraln3_lamb-400.)/(600.-400.)

                    loss_chiraln2 = torch.mean((vit_chiraln2_lamb-lamblist)**2)
                    loss_chiraln3 = torch.mean((vit_chiraln3_lamb-lamblist)**2)

                    vit_loss_weight = 1e1
                    loss += vit_loss_weight*loss_chiraln2.to(cpu)
                    loss += vit_loss_weight*loss_chiraln3.to(cpu)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.graph.parameters(), 1e1)
                self.optimizer.step()
                self.scheduler.step()

                with open(self.lossf, 'ab') as f:
                    np.savetxt(f, np.c_[global_step, loss.item()], 
                               fmt = '%d %.3e')

                if global_step % self.log_gap == 0 or global_step == 0:
                    savemodel(graph = self.graph, optim = self.optimizer, 
                              path = os.path.join(
                                  self.model_dir, 'model_step{}.pth'.format(global_step)
                                  ))

                global_step += 1

        savemodel(graph = self.graph, optim = self.optimizer, 
                  path = os.path.join(self.model_dir, 'final_model.pth'))
