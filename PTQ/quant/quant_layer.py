import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.aug_delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

        # Parameters for AugQ granularity
        self.maxi = None
        self.mini = None
        self.step = None

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                self.aug_delta, self.zero_point, self.step, self.maxi, self.mini = self.init_quantization_scale(x, self.channel_wise)
                self.delta = torch.nn.Parameter(self.aug_delta[0])   # Send to layer/block reconstruction
                # self.zero_point = torch.nn.Parameter(self.zero_point)
            else:
                self.aug_delta, self.zero_point, self.step, self.maxi, self.mini = self.init_quantization_scale(x, self.channel_wise)
                self.delta = self.aug_delta[0]   # Send to layer/block reconstruction
            self.inited = True

        # start quantization
        data_quant = torch.where((self.mini < x) & (x < self.step[0]),
                                 x.div(self.aug_delta[0]).round().add(self.zero_point[0]).clamp(0, self.n_levels - 1).sub(
                                     self.zero_point[0]).mul(self.aug_delta[0]),
                                 x
                                 )
        data_quant = torch.where((self.step[0] < x) & (x < self.step[1]),
                                 x.div(self.aug_delta[1]).round().add(self.zero_point[1]).clamp(0, self.n_levels - 1).sub(
                                     self.zero_point[1]).mul(self.aug_delta[1]),
                                 data_quant
                                 )
        data_quant = torch.where((self.step[1] < x) & (x < self.step[2]),
                                 x.div(self.aug_delta[2]).round().add(self.zero_point[2]).clamp(0, self.n_levels - 1).sub(
                                     self.zero_point[2]).mul(self.aug_delta[2]),
                                 data_quant
                                 )
        data_quant = torch.where((self.step[2] < x) & (x < self.maxi),
                                 x.div(self.aug_delta[3]).round().add(self.zero_point[3]).clamp(0, self.n_levels - 1).sub(
                                     self.zero_point[3]).mul(self.aug_delta[3]),
                                 data_quant
                                 )

        return data_quant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q, scale, zps, steps = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = scale
                        zero_point = zps
                        step = steps
                        maxi = new_max
                        mini = new_min
            else:
                raise NotImplementedError

        return delta, zero_point, step, maxi, mini

    # base quantization function
    def quantize(self, data_tensor, max, min):

        n_levels = 2 ** self.n_bits
        step = (max - min) / 4
        step_a = min + step
        step_b = step_a + step
        step_c = step_b + step

        scale_a = (step_a - min) / (n_levels - 1)
        scale_b = (step_b - step_a) / (n_levels - 1)
        scale_c = (step_c - step_b) / (n_levels - 1)
        scale_d = (max - step_c) / (n_levels - 1)

        zero_point_a = (-min / scale_a).round()
        zero_point_b = (-step_a / scale_b).round()
        zero_point_c = (-step_b / scale_c).round()
        zero_point_d = (-step_c / scale_d).round()

        data_quant = torch.where((min < data_tensor) & (data_tensor < step_a),
                                 data_tensor.div(scale_a).round().add(zero_point_a).clamp(0, n_levels - 1).sub(
                                     zero_point_a).mul(scale_a),
                                 data_tensor
                                 )
        data_quant = torch.where((step_a < data_tensor) & (data_tensor < step_b),
                                 data_tensor.div(scale_b).round().add(zero_point_b).clamp(0, n_levels - 1).sub(
                                     zero_point_b).mul(scale_b),
                                 data_quant
                                 )
        data_quant = torch.where((step_b < data_tensor) & (data_tensor < step_c),
                                 data_tensor.div(scale_c).round().add(zero_point_c).clamp(0, n_levels - 1).sub(
                                     zero_point_c).mul(scale_c),
                                 data_quant
                                 )
        data_quant = torch.where((step_c < data_tensor) & (data_tensor < max),
                                 data_tensor.div(scale_d).round().add(zero_point_d).clamp(0, n_levels - 1).sub(
                                     zero_point_d).mul(scale_d),
                                 data_quant
                                 )

        return data_quant, [scale_a, scale_b, scale_c, scale_d], [zero_point_a, zero_point_b, zero_point_c, zero_point_d], [step_a, step_b, step_c]

    # To change the bitwidth of first and last layer to 8-bits; this func is utilized.
    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
