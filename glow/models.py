import torch 
import torch.nn as nn
import numpy as np 
from . import thops
from . import modules
from . import utils 
from . import resnet 
from .transformation import RQStf
from .distributions import ConditionalDiagonalNormal

DEFAULT_TAIL_BOUND = 2.0 
DEFAULT_NUM_BINS = 10

def _RQS_params_generator(in_channels, out_channels, hidden_channels=64, num_bins = 10):
    num_params = out_channels*(3*num_bins - 1)
    return resnet.NN2dNet(
        in_features = in_channels, 
        out_features = num_params, 
        hidden_features = hidden_channels
    )

class RQS_params_generator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels = 64, num_bins = 10):
        super().__init__()
        num_params = out_channels*(3*num_bins - 1)
        self.resnet =  resnet.NN2dNet(
            in_features = in_channels, 
            out_features = num_params, 
            hidden_features = hidden_channels
            )
    def forward(self, x):
        return self.resnet(x)

class FlowStep(nn.Module):
    r"""The Flow transformation in each layer"""
    def __init__(self, in_channels, hidden_channels, 
                 actnorm_scale = 1.0):
        super().__init__()
        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale)
        # 2. permute
        self.invconv = modules.InvertibleConv1x1(in_channels, LU_decomposed = False)
        # 3. coupling 
        self.rqs_params_generator = RQS_params_generator(
                in_channels = in_channels // 2, 
                out_channels = in_channels // 2, 
                hidden_channels = hidden_channels, 
                num_bins = DEFAULT_NUM_BINS
                )
        self.rqstf = RQStf(tails = 'linear', 
                           tail_bound = DEFAULT_TAIL_BOUND, 
                           num_bins = DEFAULT_NUM_BINS)

    def forward(self, input, logdet = None, reverse = False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1. actnorm 
        z, logdet = self.actnorm(input, logdet = logdet, reverse = False)
        # 2. permute 
        z, logdet = self.invconv(z, logdet, False)
        # 3. coupling
        z1, z2 = thops.split_feature(z, 'split')
        params = self.rqs_params_generator(z1)
        z2, dlogdet = self.rqstf.forward(inputs = z2, transform_params = params, inverse = False)
        dlogdet = utils.sum_except_batch(dlogdet)
        logdet = logdet + dlogdet

        z = thops.cat_feature(z1, z2)
        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1. coupling 
        z1, z2 = thops.split_feature(input, 'split')
        params = self.rqs_params_generator(z1)
        z2, dlogdet = self.rqstf.forward(inputs = z2, transform_params=params, inverse = True)
        dlogdet = utils.sum_except_batch(dlogdet)
        logdet = logdet + dlogdet
        z = thops.cat_feature(z1, z2)
        # 2. permute
        z, logdet = self.invconv(z, logdet, True)
        # 3. actnorm
        z, logdet = self.actnorm(z, logdet = logdet, reverse = True)
        return z, logdet
        

class FlowNet(nn.Module):
    def __init__(self, 
                 image_shape, 
                 hidden_channels, 
                 K: int, 
                 L: int, 
                 actnorm_scale = 1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K 
        self.L = L
        C, H, W = image_shape

        for i in range(L):
            # 1. Squeeze 
            C, H, W = 4*C, H//2, W//2 
            self.layers.append(modules.SqueezeLayer(factor = 2))
            self.output_shapes.append([-1, C, H, W])
            # 2. Flow Transformation Layer
            for _ in range(K):
                self.layers.append(
                        FlowStep(
                            in_channels = C, 
                            hidden_channels = hidden_channels, 
                            actnorm_scale = actnorm_scale, 
                            )
                        )
                self.output_shapes.append([-1, C, H, W])
            # 3. split2d
            if i < L -1:
                self.layers.append(modules.Split2d(num_channels = C))
                self.output_shapes.append([-1, C//2, H, W])
                C = C // 2

    def forward(self, input, logdet = 0.0, reverse = False):
        if not reverse:
            return self.encode(input, logdet)
        else:
            return self.decode(input)

    def encode(self, z, logdet = 0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse = False)
        return z, logdet
    
    def decode(self, z):
        for layer in reversed(self.layers):
            if isinstance(layer, modules.Split2d):
                z, logdet = layer(z, logdet = 0.0, reverse = True, eps_std = None)
            else:
                z, logdet = layer(z, logdet = 0.0, reverse = True)
        return z

class Glow(nn.Module):
    def __init__(self, image_shape, 
                 num_labels, 
                 hidden_channels, 
                 K, L, 
                 actnorm_scale = 0.1, 
                 num_context = 128):
        super().__init__()
        self.flow = FlowNet(image_shape = image_shape, 
                            hidden_channels = hidden_channels, 
                            K = K, L = L, 
                            actnorm_scale = actnorm_scale)
        self.num_labels = num_labels

        self.num_channels = self.flow.output_shapes[-1][1]
        self.h = self.flow.output_shapes[-1][2]
        self.w = self.flow.output_shapes[-1][3]
        self.num_context = num_context
        self.inputs_encoder = resnet.ResidualNet(
                in_features = self.num_labels, 
                out_features = self.num_context, 
                hidden_features = self.num_context, 
                num_blocks = 3
                )
        self.context_encoder = nn.Linear(
                self.num_context, 
                2*self.num_channels*self.h*self.w)

        self.context_encoder.weight.data.normal_(0.0, 1e-4)
        self.context_encoder.bias.data.normal_(0.0, 1e-4)
        self.distribution = ConditionalDiagonalNormal(
                shape = [self.num_channels, self.h, self.w], 
                context_encoder = self.context_encoder
                )

    def forward(self, x= None, y_onehot = None, z = None, 
                reverse = False, num_samples = 2):
        if not reverse:
            return self.normal_flow(x, y_onehot)
        else:
            return self.reverse_flow(
                    y_onehot = y_onehot, num_samples = num_samples)

    def prior(self, noise, y_onehot = None):
        label_context = self.inputs_encoder(y_onehot)
        log_prob_noise = self.distribution.log_prob(
                noise, context = label_context)
        return log_prob_noise
        
    def normal_flow(self, x, y_onehot):
        pixels = thops.pixels(x)
        z = x
        logdet = 0.0
        # encode
        z, objective = self.flow(z, logdet = logdet, reverse = False)
        objective += self.prior(noise = z, y_onehot = y_onehot)

        nll = (-objective) / float(np.log(2.0)*pixels)
        return z, nll

    def reverse_flow(self, y_onehot, num_samples = 2):
        label_context = self.inputs_encoder(y_onehot)
        z = self.distribution.sample(num_samples, context = label_context)
        noise = utils.merge_leading_dims(x = z, num_dims = 2)
        x = self.flow(noise, reverse = True)
        x = utils.split_leading_dim(x = x, shape = [-1, num_samples])
        return x

    def latent_polyinterp(self, basex, basey, lamblist, num_samples = 64):
        r"""
        generate samples based on existed samples and their corresponding 
        labels.

        inputs: 
        - basex: the existed samples.
        - basey: the labels corresponding to the existed samples.
        - lamblist: the diresed labels for the generated samples.
        - num_samples: the number of generated samples for each label.

        outputs:
        - x: the generated samples.
        """
        # tranfer samples x to latent space z = Glow(x)
        zlist  = basex
        logdet = torch.zeros_like(basex[:, 0, 0, 0])
        zlist, _ = self.flow(zlist, logdet = logdet, reverse = False)
        Bz, C, H, W = zlist.shape

        # calculate z(lambs) using interpolation
        targetz = torch.zeros(len(lamblist), num_samples, C, H, W)
        for ilamb, lamb in enumerate(lamblist):
            p = 0
            for iz, z in enumerate(zlist):
                baselamb = basey[basey!=basey[iz]]
                p1 = lamb - baselamb
                p2 = basey[iz] - baselamb
                p += (torch.prod(p1) / torch.prod(p2))*z
            targetz[ilamb, :] = p
        noise = utils.merge_leading_dims(x = targetz, num_dims = 2)

        # transfer z back to x = Glow^{-1}(z)
        x = self.flow(noise, reverse = True)
        x = utils.split_leading_dim(x = x, shape = [-1, num_samples])
        x = torch.mean(x, dim = 1)
        return x

    def interp_sample_generator(
            self, y_train_list, y_pred, labelid, num_samples = 64
            ):
        r""" generate samples based on the labels used for training. 

        inputs: 
        - y_train_list: the labels used to trained Glow. 
        - y_pred: the desired samples' labels.
        - labelid: the index of the desired label for multiple-label dataset.
        - num_samples: the number of generated samples for the desired samples.

        outputs:
        - x: the generated samples.
        """
        # generate z in latent space
        label_context = self.inputs_encoder(y_train_list)
        zlist = self.distribution.sample(num_samples, context = label_context)
        NL, NB, C, H, W = zlist.shape

        y_train_id = y_train_list[:, labelid]
        y_pred_id = y_pred[:, labelid]

        targetz = torch.zeros(num_samples, C, H, W)
        for isample in range(NB):
            p = 0 
            for iz in range(NL):
                z = zlist[iz, isample]
                baselamb = y_train_id[y_train_id != y_train_id[iz]]
                p1 = y_pred_id - baselamb
                p2 = y_pred_id[iz] - baselamb
                p += (torch.prod(p1)/torch.prod(p2)) * z
            targetz[isample] = p
        x = self.flow(targetz, reverse = True)
        return x

    def predsample(self, y_pred, num_samples = 3):
        r"""
        generate samples for desired labels. 

        inputs:
        - y_pred: the desired labels. 
        - num_samples: the number of generated samples for each label.

        outputs:
        - x: the generated samples.
        """
        label_context = self.inputs_encoder(y_pred)
        zlist = self.distribution.sample(num_samples, 
                                         context = label_context)
        noise = utils.merge_leading_dims(x = zlist, num_dims = 2)
        x = self.flow(noise, reverse = True)
        x = utils.split_leading_dim(x = x, shape = [-1, num_samples])
        return x

    @staticmethod 
    def loss_generative(nll):
        return torch.mean(nll)
