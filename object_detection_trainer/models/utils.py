#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AI Trainer - Model Utilities
================================================================================

 Based on fasterrcnn-pytorch-training-pipeline by Sovit Ranjan Rath
 https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
 Original code: Copyright (c) 2023 Sovit Ranjan Rath, MIT License
 (see object_detection_trainer/LICENSE).

 Modifications: Jimmy Majumder (j.majumder@qibitech.com), 2024_Jun
 Modifications Copyright (c) 2024 QibiTech Inc.

 NOTICE:
 This code and data is for research and testing purposes only.
 DO NOT use for product development or production phase.
 For commercial use or inquiries, contact: j.majumder@qibitech.com

================================================================================
"""

import torch

import torch.distributed as dist
from torch.autograd.function import Function

BatchNorm2d = torch.nn.BatchNorm2d
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
    )
        
def differentiable_all_reduce(input: torch.Tensor) -> torch.Tensor:
    """
    Differentiable counterpart of `dist.all_reduce`.
    """
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return input
    return _AllReduce.apply(input)

class _AllReduce(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(grad_output, async_op=False)
        return grad_output