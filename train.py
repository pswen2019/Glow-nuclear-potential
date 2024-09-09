import os 
import vision 
import argparse
import torch 
import json 
import datetime
import numpy as np

from glow.trainer import Trainer
from glow.models import Glow
from glow.utils import loadmodel
from vitmodel import ViT, loadvitmodel

cwd = os.getcwd()
date = datetime.datetime.now().strftime("Glow_%Y_%m_%d_%H_%M_%S_%f")
defaultlogdir = os.path.join(cwd, 'result', date)

###############################################################################
# cmd parsers
parser = argparse.ArgumentParser()
# dataset related
parser.add_argument('--dataset', type = str, default = 'chiral', 
                    help = 'Dataloader for samples.')
parser.add_argument('--datasetdir', type = str, 
                    default = os.path.join(cwd, 'chiralpot'), 
                    help = 'the directory saving potential samples')
parser.add_argument('--batchsize', type = int, default = 6, 
                    help = 'batch size of the training for each epoch')
parser.add_argument('--imgsize', type = int, default = 48, 
                    help = 'number of mesh points of potential matrix elements')
parser.add_argument('--imgchannel', type = int, default = 14, 
                    help = 'number of partial wave channels in potential samples')
parser.add_argument('--numlabel', type = int, default = 2, 
                    help = 'number of labels of potential')

# Glow model related
parser.add_argument('--K', type = int, default = 4)
parser.add_argument('--L', type = int, default = 4)
parser.add_argument('--hchannels', type = int, default = 512, 
                    help = 'the number of hidden channels for RQS transformation')
parser.add_argument('--hcontext', type = int, default = 64, 
                    help = 'the number of hidden context for latent distribution')
parser.add_argument('--actnorm', type = float, default = 1.0, 
                    help = 'the scale for ActNorm transformation')
parser.add_argument('--model', type = str, default = None, 
                    help = 'the path to trained model')
parser.add_argument('--modelconfig', type = str, default = None, 
                    help = 'the path to the trained model config')

# training related 
parser.add_argument('--num_epoch', type = int, default = 1000, 
                    help = 'number of training epoch')
parser.add_argument('--lr', type = float, default = 1e-4, 
                    help = 'learning rate')
parser.add_argument('--logdir', type = str, default = defaultlogdir, 
                    help = 'the directory saving trained models')
parser.add_argument('--loggap', type = int, default = 100, 
                    help = 'saving trained models for every loggap steps')
parser.add_argument('--device', type = str, default = 'cuda', 
                    help = 'the device training Glow model')

# ViT model related
parser.add_argument('--vitchiraln2model', type = str, default = None, 
                    help = 'the path to trained ViT model for chiral v=2 potentials')
parser.add_argument('--vitchiraln2config', type = str, default = None, 
                    help = 'the path to trained ViT model config for chiral v=2 potentials')
parser.add_argument('--vitchiraln2normc', type = str, default = None, 
                    help = 'the path to trained ViT model normalizing constants for chiral v=2 potentials')
parser.add_argument('--vitchiraln3model', type = str, default = None, 
                    help = 'the path to trained ViT model for chiral v=3 potentials')
parser.add_argument('--vitchiraln3config', type = str, default = None, 
                    help = 'the path to trained ViT model config for chiral v=3 potentials')
parser.add_argument('--vitchiraln3normc', type = str, default = None, 
                    help = 'the path to trained ViT model normalizing constants for chiral v=3 potentials')
parser.add_argument('--vitdevice', type = str, default = 'cuda', 
                    help = 'the device for ViT models')
args = parser.parse_args()
config = vars(args)

if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

# save arg config
configf = os.path.join(args.logdir, 'config.json')
open(configf, 'w').close()
if args.model is not None:
    with open(args.modelconfig, 'r') as f:
        modelconfig = json.load(f)
    config['dataset'] = modelconfig['dataset']
    config['imgchannel'] = modelconfig['imgchannel']
    config['imgsize'] = modelconfig['imgsize']
    config['numlabel'] = modelconfig['numlabel']
    config['hchannels'] = modelconfig['hchannels']
    config['hcontext'] = modelconfig['hcontext']
    config['actnorm'] = modelconfig['actnorm']
    config['K'] = modelconfig['K']
    config['L'] = modelconfig['L']
with open(configf, 'w') as f:
    json.dump(config, f, indent = 2)

dataset = vision.Datasets[config['dataset']](root_dir = config['datasetdir'])
C, H, W = dataset[0]['x'].shape
num_label = len(dataset[0]['y_onehot']) 
print(dataset[0]['x'].shape)
print(dataset[0]['y_onehot'])

# create (and load pretrained) Glow model
graph = Glow(
        image_shape = (config['imgchannel'], config['imgsize'], config['imgsize']),
        K = config['K'], L = config['L'],
        num_labels = config['numlabel'], 
        hidden_channels = config['hchannels'], 
        num_context = config['hcontext'], 
        actnorm_scale = config['actnorm']
        )
if config['model'] is not None:
    graph, _ = loadmodel(path = config['model'], graph = graph)

# create (and load pretrained) ViT models
if config['vitchiraln2model'] is not None and config['vitchiraln3model'] is not None:
    vitdevice = torch.device(config['vitdevice'])

    # create and load ViT model for chiral v = 2
    with open(config['vitchiraln2config'], 'r') as f:
        vit_chiraln2_config = json.load(f)
    vit_chiraln2 = ViT(
            image_size  = vit_chiraln2_config['imgsize'], 
            patch_size  = vit_chiraln2_config['patch_size'], 
            num_classes = vit_chiraln2_config['nclass'], 
            dim         = vit_chiraln2_config['dim'], 
            depth       = vit_chiraln2_config['depth'], 
            heads       = vit_chiraln2_config['heads'], 
            mlp_dim     = vit_chiraln2_config['mlpdim'], 
            channels    = vit_chiraln2_config['channels'], 
            pool        = vit_chiraln2_config['pool'])
    vit_chiraln2 = loadvitmodel(
            vit_chiraln2, 
            path = config['vitchiraln2model'], device = vitdevice)
    vit_chiraln2_normc = torch.from_numpy(
            np.loadtxt(config['vitchiraln2normc'])).to(vitdevice)

    # create and load ViT model for chiral v = 3
    with open(config['vitchiraln3config'], 'r') as f:
        vit_chiraln3_config = json.load(f)
    vit_chiraln3 = ViT(
            image_size  = vit_chiraln3_config['imgsize'], 
            patch_size  = vit_chiraln3_config['patch_size'], 
            num_classes = vit_chiraln3_config['nclass'], 
            dim         = vit_chiraln3_config['dim'], 
            depth       = vit_chiraln3_config['depth'], 
            heads       = vit_chiraln3_config['heads'], 
            mlp_dim     = vit_chiraln3_config['mlpdim'], 
            channels    = vit_chiraln3_config['channels'], 
            pool        = vit_chiraln3_config['pool'])
    vit_chiraln3 = loadvitmodel(
            vit_chiraln3, 
            path = config['vitchiraln3model'], device = vitdevice)
    vit_chiraln3_normc = torch.from_numpy(
            np.loadtxt(config['vitchiraln3normc'])).to(vitdevice)
else:
    vit_chiraln2 = None
    vit_chiraln3 = None
    vit_chiraln2_normc = None
    vit_chiraln3_normc = None

trainer = Trainer(
        graph = graph, 
        dataset = dataset, 
        device = config['device'], 
        num_epoch = config['num_epoch'], 
        lr = config['lr'], 
        log_dir = config['logdir'], 
        batch_size = config['batchsize'], 
        vit_chiraln2 = vit_chiraln2, 
        vit_chiraln2_normc = vit_chiraln2_normc, 
        vit_chiraln3 = vit_chiraln3, 
        vit_chiraln3_normc = vit_chiraln3_normc, 
        vit_device = config['vitdevice'], 
        log_gap = config['loggap']
        )
trainer.train()
