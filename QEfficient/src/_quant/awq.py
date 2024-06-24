# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from dataclasses import dataclass

from QEfficient.src._transformers.auto import QEFFAutoModelForCausalLM


@dataclass
class AWQQuantConfig:
    pass


def get_quant_config_from_pretrained_model_path(pretrained_model_path: str):
    quant_config_file = None
    
    if os.path.isfile(os.path.join(pretrained_model_path, "quant_config.json")):
        quant_config_file = os.path.join(pretrained_model_path, "quant_config.json")
    elif os.path.isfile(os.path.join(pretrained_model_path, "quantize_config.json")):
        quant_config_file = os.path.join(pretrained_model_path, "quant_config.json")

    if quant_config_file is not None:
        quant_config = json.load(open(quant_config_file))
        return quant_config
    

class QEFFAWQModelForCausalLM(QEFFAutoModelForCausalLM):
    def from_pretrained(self, pretrained_model_name_or_path: str, *args, **kwargs):
        transform: bool = kwargs.get("transform", True)
        kwargs.update({"use_cache": True})  # Always pass use_cache = True, to get KV values as output during ONNX export 
        
        """
        Make sure that awq package should not be required if user doesn't want to load any AWQ model, so keep import statements within this.
        1. Load the model weights autoAWQ style instead of qllm style
        2. Check if model is supported by initializing AWQQuantConfig
        3. load quantized modules autoAWQ style
        4. repack the weights qllm style
        5. return
        """
        pass