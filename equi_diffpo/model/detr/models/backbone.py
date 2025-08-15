# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from ..util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed
from escnn import gspaces, nn as enn
import torch

class EquiResBlock(torch.nn.Module):
    def __init__(
        self,
        group: gspaces.GSpace2D,
        input_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        initialize: bool = True,
    ):
        super(EquiResBlock, self).__init__()
        self.group = group
        rep = self.group.regular_repr

        feat_type_in = enn.FieldType(self.group, input_channels * [rep])
        feat_type_hid = enn.FieldType(self.group, hidden_dim * [rep])

        self.layer1 = enn.SequentialModule(
            enn.R2Conv(
                feat_type_in,
                feat_type_hid,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                initialize=initialize,
            ),
            enn.ReLU(feat_type_hid, inplace=True),
        )

        self.layer2 = enn.SequentialModule(
            enn.R2Conv(
                feat_type_hid,
                feat_type_hid,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                initialize=initialize,
            ),
        )
        self.relu = enn.ReLU(feat_type_hid, inplace=True)

        self.upscale = None
        if input_channels != hidden_dim or stride != 1:
            self.upscale = enn.SequentialModule(
                enn.R2Conv(feat_type_in, feat_type_hid, kernel_size=1, stride=stride, bias=False, initialize=initialize),
            )

    def forward(self, xx: enn.GeometricTensor) -> enn.GeometricTensor:
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)

        return out
    

class EquivariantResEncoder76Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 2, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 84x84 -> 80x80
            enn.R2Conv(
                enn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                enn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                kernel_size=5,
                padding=0,
                initialize=initialize,
            ),
            # 80x80
            enn.ReLU(enn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=True),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=True),
            enn.PointwiseMaxPool(enn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 40x40
            EquiResBlock(self.group, n_out // 8, n_out // 4, initialize=True),
            EquiResBlock(self.group, n_out // 4, n_out // 4, initialize=True),
            enn.PointwiseMaxPool(enn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 20x20
            EquiResBlock(self.group, n_out // 4, n_out // 2, initialize=True),
            EquiResBlock(self.group, n_out // 2, n_out // 2, initialize=True),
            enn.PointwiseMaxPool(enn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 10x10
            EquiResBlock(self.group, n_out // 2, n_out, initialize=True),
            EquiResBlock(self.group, n_out, n_out, initialize=True),
            enn.PointwiseMaxPool(enn.FieldType(self.group, n_out * [self.group.regular_repr]), 3),
            # 3x3
            enn.R2Conv(
                enn.FieldType(self.group, n_out * [self.group.regular_repr]),
                enn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=1,  # Changed from 0 to 1 to maintain 3x3 output
                initialize=initialize,
            ),
            enn.ReLU(enn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 3x3 (preserved)
        )

    def forward(self, x) -> enn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = enn.GeometricTensor(x, enn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class EquivariantBackbone(nn.Module):
    """Equivariant backbone using EquivariantResEncoder76Cyclic."""
    def __init__(self, obs_channel: int = 3, n_out: int = 128, N: int = 8):
        super().__init__()
        self.backbone = EquivariantResEncoder76Cyclic(obs_channel=obs_channel, n_out=n_out, N=N, initialize=True)
        self.num_channels = n_out * N
        print("num_channels", self.num_channels)

    def forward(self, tensor):
        # Extract tensor from NestedTensor if needed
        if hasattr(tensor, 'tensors'):
            x = tensor.tensors
        else:
            x = tensor
            
        # Forward through equivariant encoder
        geometric_tensor = self.backbone(x)
        
        # Convert GeometricTensor back to regular tensor
        # The tensor contains all rotation channels
        output_tensor = geometric_tensor.tensor
        
        # Return in the expected format for DETR
        return {"0": output_tensor}


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build_backbone_equi(args):
    position_embedding = build_position_encoding(args)
    
    # Default parameters for EquivariantResEncoder76Cyclic
    obs_channel = getattr(args, 'obs_channel', 3)  # RGB channels by default
    N = getattr(args, 'N', 8)                 # Number of rotations in cyclic group
    n_out = 512 // N     # Output feature dimension
    print(n_out, obs_channel, N)
    backbone = EquivariantBackbone(obs_channel=obs_channel, n_out=n_out, N=N)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
