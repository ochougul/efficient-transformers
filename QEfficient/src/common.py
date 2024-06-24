# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP dictionary defines the mapping between names of the varities of Transformer model defined in 
QEFF_MODEL_TYPE and the classes that implement the methods i.e.(compile, export etc.) for those types.

QEFFAutoModel provides a common interface for loading the HuggingFace models using either the HF card name of local path of downloaded model.
"""

import os
from enum import Enum
from typing import Any, Dict, Type

import transformers
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

from QEfficient.src._transformers.auto import QEFFAutoModelForCausalLM
from QEfficient.src.base import QEFFBaseModel
from QEfficient.utils._utils import login_and_download_hf_lm


class QEFF_MODEL_TYPE(Enum):
    """
    Defines Names of the different varities of transformer models.
    """
    CAUSALLM = "LLM"
    DIFFUSION = "DIFFUSION"
    AWQ = "AWQ"


MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP: Dict[QEFF_MODEL_TYPE, Type[QEFFBaseModel]] = {
    QEFF_MODEL_TYPE.CAUSALLM: QEFFAutoModelForCausalLM
}

AUTO_MODEL_MAP_TO_MODEL_TYPE_MAP: Dict[Type[QEFFBaseModel], QEFF_MODEL_TYPE] = {v:k for k,v in MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP.items()}


def check_if_model_is_awq(config) -> bool:
    """
    Relying on the quantization_config/quant_config attribute in `config.json` of the model to identify if the model
    is quantized and the if it's AWQ/GPTQ etc.
    """
    quant_config = getattr(config, "quantization_config", getattr(config, "quant_config", None))
    
    if quant_config is None:
        return False  # Model is not quantized
    
    if quant_config.get("quant_method", None) == "awq":
        return True  # Only AWQ is supported as of now.
    else:
        raise NotImplementedError(f"current model type is not yet supported {type(config)}")


def get_hf_model_type(hf_model_path: str) -> QEFF_MODEL_TYPE:
    """
    Loads model config file and returns the type of the model (i.e. LLMs, SD, quantized etc.) as supported by the library.
    """
    assert os.path.isdir(hf_model_path), "Pleae pass local dir path where the model is downloaded; use `QEfficient.utils.login_and_download_hf_lm` for downloading hf model"
    config, kwargs = AutoConfig.from_pretrained(
                hf_model_path,
                return_unused_kwargs=True,
            )
    assert config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING, f"Could not find {type(config.__class__)} in {MODEL_FOR_CAUSAL_LM_MAPPING.keys()} for transformers=={transformers.__version__}"
    
    if check_if_model_is_awq(config):
        return QEFF_MODEL_TYPE.AWQ
    else:
        return QEFF_MODEL_TYPE.CAUSALLM


class QEFFCommonLoader:
    """
    Provides HuggingFace model loading interface same as transformers APIs.
    Supports loading any model on HuggingFace.
    """
    def __init__(self, *args: Any, **kwds: Any) -> None:
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)`")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> QEFFBaseModel:
        """
        Downloads HuggingFace model if already doesn't exist locally, returns QEffAutoModel object based on type of model.
        """
        # Download model if already doesn't exist
        pretrained_model_name_or_path = pretrained_model_name_or_path if os.path.isdir(pretrained_model_name_or_path) \
            else login_and_download_hf_lm(pretrained_model_name_or_path, *args, **kwargs)
        
        # Get QEFF Class from type of the model
        model_type = get_hf_model_type(hf_model_path=pretrained_model_name_or_path)
        qeff_auto_model_class = MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP[model_type]
        assert issubclass(qeff_auto_model_class, QEFFBaseModel), f"Expected class that inherits {QEFFBaseModel}, got {type(qeff_auto_model_class)}"
        
        # Initialize QEFF model class
        return qeff_auto_model_class.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
