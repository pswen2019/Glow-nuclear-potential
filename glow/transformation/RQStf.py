from .splines import rational_quadratic_spline
from .splines import unconstrained_rational_quadratic_spline
from .splines import DEFAULT_MIN_DERIVATIVE, DEFAULT_MIN_BIN_WIDTH, DEFAULT_MIN_BIN_HEIGHT
import torch
import torch.nn as nn
# import torch.nn.functional as F


class RQStf():
    def __init__(self, 
                 num_bins = 10, 
                 tails = None, 
                 tail_bound = 1., 
                 min_bin_width = DEFAULT_MIN_BIN_WIDTH, 
                 min_bin_height = DEFAULT_MIN_BIN_HEIGHT, 
                 min_derivative = DEFAULT_MIN_DERIVATIVE):
        self.num_bins = num_bins 
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

    def forward(self, 
                inputs, 
                transform_params, 
                inverse=False):

        if inputs.dim() == 4:
            b, c, h, w = inputs.shape
            tf_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        elif inputs.dim() == 2:
            b, d = inputs.shape
            tf_params = transform_params.reshape(b, d, -1)
        
        outputs, logabsdet = self._transform_forward(
            inputs = inputs, 
            transform_params = tf_params, 
            inverse = inverse
            )

        return outputs, logabsdet
    
    def _transform_forward(self, 
                           inputs, 
                           transform_params, 
                           inverse = False):

        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2*self.num_bins]
        unnormalized_derivatives = transform_params[..., 2*self.num_bins:]
        
        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {
                "tails": self.tails, 
                "tail_bound": self.tail_bound
            }
        return spline_fn(
            inputs = inputs, 
            unnormalized_widths = unnormalized_widths, 
            unnormalized_heights = unnormalized_heights, 
            unnormalized_derivatives = unnormalized_derivatives, 
            inverse = inverse, 
            **spline_kwargs
        )
