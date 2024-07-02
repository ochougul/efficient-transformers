# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import gc
import json
import os

import torch
import torch.nn as nn
import tqdm
from transformers import AutoConfig

from QEfficient.src._transformers.auto import QEFFAutoModelForCausalLM


def get_quant_config_from_pretrained_model_path(pretrained_model_path: str):
    quant_config_file = None
    
    if os.path.isfile(os.path.join(pretrained_model_path, "quant_config.json")):
        quant_config_file = os.path.join(pretrained_model_path, "quant_config.json")
    elif os.path.isfile(os.path.join(pretrained_model_path, "quantize_config.json")):
        quant_config_file = os.path.join(pretrained_model_path, "quant_config.json")

    if quant_config_file is not None:
        quant_config = json.load(open(quant_config_file))
        return quant_config
    

def check_if_awq_model_is_supported(pretrained_model_name_or_path, **kwargs):
    if kwargs.get("force_download", None) is None:
        kwargs.update({"force_download": False})
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    quant_config = getattr(config, "quantization_config", getattr(config, "quant_config", None))
    assert quant_config is not None, "Expected quantization config to be present in config.json"
    assert quant_config.get("version", None).lower() == "gemm", f"Only gemm version AWQ models is supported as of now got {quant_config.version}"


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):  # noqa:B006
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():  # noqa:SIM108
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def repack_transform(model: nn.Module):
    from awq.modules.linear import WQLinear_GEMM
    from QEfficient.src._quant.quant_linear_onnxruntime import QuantLinearORT
    from QEfficient.src._quant.compress_weight import unpack_wqlinear_gemm
    source_layer = WQLinear_GEMM
    target_layer = QuantLinearORT
    qlayers = find_layers(model, [source_layer])

    for module_name, qlayer in tqdm.tqdm(qlayers.items(),
            desc="repacking model from pack_mode=`GEMM` to `ORT`"):
        fp16_weight, scales, zeros = unpack_wqlinear_gemm(qlayer)
        qlayer.weight = fp16_weight
        tmp = qlayer
        new_module = target_layer(tmp.w_bit,tmp.group_size, tmp.in_features, tmp.out_features, tmp.bias is not None)
        set_op_by_name(model, module_name, new_module)
        new_module.pack(tmp, scales.T, zeros.T, torch.tensor([i // tmp.group_size for i in range(tmp.in_features)], dtype=torch.int32))

    del qlayers
    gc.collect()
    return model


class QEFFAWQModelForCausalLM(QEFFAutoModelForCausalLM):
    @classmethod
    def from_quantized(cls, pretrained_model_name_or_path: str, /,  **kwargs):
        """
        Make sure that awq package should not be required if user doesn't want to load any AWQ model, so keep import statements within this.
        1. Load the model weights autoAWQ style instead of qllm style
        2. Check if model is supported by initializing AWQQuantConfig
        3. load quantized modules autoAWQ style
        4. repack the weights qllm style
        5. return
        FIXME: IT is possible to add from_quantized method in QEFFAutoModelForCausalLM -> think if that's a better option?
        """
        from awq import AutoAWQForCausalLM
        check_if_awq_model_is_supported(pretrained_model_name_or_path, **kwargs)
        model_with_replaced_linear_layers = AutoAWQForCausalLM.from_quantized(pretrained_model_name_or_path, fuse_layers=False, **kwargs)
        import ipdb
        ipdb.set_trace()
        repack_transform(model_with_replaced_linear_layers)
        return cls(model_with_replaced_linear_layers)
