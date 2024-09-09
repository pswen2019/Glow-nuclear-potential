"""Various PyTorch utility functions."""

import torch

from . import typechecks as typechecker

import matplotlib.pyplot as plt

import os

import numpy as np

def tile(x, n):
    if not typechecker.is_positive_int(n):
        raise TypeError('Argument \'n\' must be a positive integer.')
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not typechecker.is_nonnegative_int(num_batch_dims):
        raise TypeError('Number of batch dimensions must be a non-negative integer.')
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if not typechecker.is_positive_int(num_dims):
        raise TypeError('Number of leading dims must be a positive integer.')
    if num_dims > x.dim():
        raise ValueError('Number of leading dims can\'t be greater than total number of dims.')
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not typechecker.is_positive_int(num_reps):
        raise TypeError('Number of repetitions must be a positive integer.')
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def tensor2numpy(x):
    return x.detach().cpu().numpy()


def logabsdet(x):
    """Returns the log absolute determinant of square matrix x."""
    # Note: torch.logdet() only works for positive determinant.
    _, res = torch.slogdet(x)
    return res


def random_orthogonal(size):
    """
    Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(size, size)
    q, _ = torch.qr(x)
    return q


def get_num_parameters(model):
    """
    Returns the number of trainable parameters in a model of type nn.Module
    :param model: nn.Module containing trainable parameters
    :return: number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters

def searchsorted(bin_locations, inputs, eps=1e-8):
    bin_locations[..., -1] += eps
    num_bins = bin_locations.size()[-1] - 1
    re = torch.sum(inputs[..., None] >= bin_locations, dim = -1) - 1
    if re.max() >= num_bins:
        print("     Warning: samples outside the integration region.")
        re[re>=num_bins] = num_bins - 1
    re[re<0] = 0
    return re

def cbrt(x):
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)