# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
RMS Norm CustomOp Node in com.qti.aisw.onnx Domain for Cloud AI 100
This is to handle the FP16 Overflow seen in RMS Norm for LLMs
"""

import onnxscript
import torch
from onnxscript.onnx_opset import opset13 as op
from torch import nn

opset_version = 13
custom_opset = onnxscript.values.Opset(domain="com.qti.aisw.onnx", version=1)


# Version 1
# @onnxscript.script(custom_opset)
# def CustomRMSNorm(hidden_states, weight, epsilon):
#     weight = op.Cast(weight, to=1)
#     variance = op.ReduceMean(op.Pow(hidden_states, 2), axes=[-1], keepdims=1)
#     epsilon = op.Constant(value_floats=[1e-5])
#     epsilon = op.Expand(epsilon, op.Shape(variance))
#     hidden_states = hidden_states * op.Reciprocal(op.Sqrt(variance + epsilon))
#     return weight * hidden_states

@onnxscript.script(custom_opset)
def CustomRMSNorm(hidden_states, weight):
    #weight = ops.Cast(weight, to=1)
    #hidden_states = ops.Cast(hidden_states, to=1)
    # hidden_states = ops.Cast(hidden_states, to=1)
    weight = op.Cast(weight, to=1)
    variance = op.ReduceMean(op.Pow(hidden_states, 2), axes=[-1], keepdims=1)
    # epsilon = op.Expand(epsilon, op.Shape(variance))
    epsilon = op.Constant(value_floats=[1e-5])
    epsilon = op.Cast(epsilon, to=1)
    hidden_states = hidden_states * op.Reciprocal(op.Sqrt(variance + epsilon))
    out = weight * hidden_states
    return op.Cast(out, to=1)


class CustomRMSNormOp(torch.autograd.Function):
    @staticmethod
    def forward(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.onnx._internal.jit_utils.GraphContext,
        hidden_states: torch.Value,
        weight: torch.Value,
        epsilon: torch.Value,
    ) -> torch.Value:
        return g.onnxscript_op(CustomRMSNorm, hidden_states, weight)


class CustomRMSNormAIC(nn.Module):
    def __init__(self, hidden_size, eps=1e-05):
        super(CustomRMSNormAIC, self).__init__()
        self.variance_epsilon = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size).float())

    def forward(self, hidden_states):
        # import ipdb
        # ipdb.set_trace()
        output = CustomRMSNormOp.apply(hidden_states, self.weight.float(), torch.Tensor([self.variance_epsilon]).float())
        return output
