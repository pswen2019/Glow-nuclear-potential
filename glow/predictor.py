import torch 
import os 
import datetime 
import numpy as np 
from tqdm import tqdm
from .utils import sympot
from .utils import merge_leading_dims

script_dir = os.path.dirname(os.path.realpath(__file__))
hbarc = 197.3
cpu = torch.device('cpu')

class Predictor():
    def __init__(self, graph, device = 'cpu', vit_device = 'cpu', 
                 log_dir = None, 
                 vit_chiraln2 = None, 
                 vit_chiraln2_normc = None, 
                 vit_chiraln3 = None, 
                 vit_chiraln3_normc = None):

        date = str(datetime)
        date = date[:date.rfind(':')].replace('-','')\
                .replace(':','').replace(' ','')
        self.log_dir = log_dir if log_dir is not None \
                else os.path.join('result', 'log_'+date)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.device = torch.device(device)
        self.graph = graph.to(self.device)

        if (vit_chiraln2 is not None) and (vit_chiraln3 is not None):
            self.vit_device = torch.device(vit_device)
            self.vit_chiraln2 = vit_chiraln2.to(vit_device)
            self.vit_chiraln3 = vit_chiraln3.to(vit_device)
            self.vit_chiraln2_scale = vit_chiraln2_normc[0]
            self.vit_chiraln2_shift = vit_chiraln2_normc[1]
            self.vit_chiraln3_scale = vit_chiraln3_normc[0]
            self.vit_chiraln3_shift = vit_chiraln3_normc[1]
            self.vit = True
            self.vit_chiraln2_logf = os.path.join(self.log_dir, 'vitlec_chiraln2.dat')
            self.vit_chiraln3_logf = os.path.join(self.log_dir, 'vitlec_chiraln3.dat')
            open(self.vit_chiraln2_logf, 'w').close()
            open(self.vit_chiraln3_logf, 'w').close()
        else:
            self.vit = False

    def predict(self, num_samples, batch_size = 10):
        batch_size = num_samples if batch_size is None else batch_size
        epoch = num_samples//batch_size
        self.graph.zero_grad()
        self.graph.eval()

        lamblist = torch.arange(450.0, 551.0, 10.0)
        lamblist = (lamblist - 400.)/(600.-400.)
        pred_label = torch.zeros(2*len(lamblist), 2)
        pred_label[:len(lamblist), 0] = 2.0
        pred_label[len(lamblist):, 0] = 3.0
        pred_label[:len(lamblist), 1] = lamblist
        pred_label[len(lamblist):, 1] = lamblist
        print('generate sample at (v, lambda)')
        print(pred_label.clone().detach().to(cpu).numpy())

        for iepoch in tqdm(range(epoch)):
            pred_samples = self.graph.predsample(
                    y_pred = pred_label.to(self.device), num_samples = batch_size
                    )
            num_labels, num_predsamples, C, H, W = pred_samples.shape
            for ilabel in range(num_labels):
                pred_samples[ilabel] = sympot(pred_samples[ilabel])

            pred_samples = pred_samples.detach().to(cpu)
            pred_label[:, 1] = 200.0*pred_label[:, 1] + 400.0
            for ilabel, labels in enumerate(pred_label):
                lfname = 'GlowPot'
                lfname += '_n{:d}lo{:d}'.format( int(labels[0].item()), int(labels[1].item()) )
                for isample in range(num_predsamples):
                    fname = lfname + '_{:d}{:d}.dat'.format(iepoch, isample)
                    fname = os.path.join(self.log_dir, fname)
                    pot = pred_samples[ilabel, isample].reshape((C*H, W))
                    with open(fname, 'ab') as f:
                        np.savetxt(f, pot.numpy(), fmt = '%.4e')

            if self.vit is not False:
                pred_samples = merge_leading_dims(pred_samples, num_dims = 2).to(self.vit_device)
                pred_chiraln2 = pred_samples[:len(pred_samples)//2]
                pred_chiraln3 = pred_samples[len(pred_samples)//2:]

                vit_chiraln2_leclamb = self.vit_chiraln2(pred_chiraln2[:, :, ::2, ::2])
                vit_chiraln3_leclamb = self.vit_chiraln3(pred_chiraln3[:, :, ::2, ::2])
                vit_chiraln2_leclamb = (vit_chiraln2_leclamb - self.vit_chiraln2_shift)/self.vit_chiraln2_scale
                vit_chiraln3_leclamb = (vit_chiraln3_leclamb - self.vit_chiraln3_shift)/self.vit_chiraln3_scale
                with open(self.vit_chiraln2_logf, 'ab') as f:
                    np.savetxt(f, vit_chiraln2_leclamb.detach().to(cpu).numpy(), fmt = '%.4e')
                with open(self.vit_chiraln3_logf, 'ab') as f:
                    np.savetxt(f, vit_chiraln3_leclamb.detach().to(cpu).numpy(), fmt = '%.4e')
