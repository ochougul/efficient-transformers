# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from typing import List, Optional, Tuple, Union

import requests
from huggingface_hub import login, snapshot_download
from requests.exceptions import HTTPError
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.utils.constants import QEFF_MODELS_DIR


def login_and_download_hf_lm(pretrained_model_name_or_path, *args, **kwargs):
    hf_token = kwargs.pop("hf_token", None)
    cache_dir = kwargs.pop("cache_dir", None)   
    if hf_token is not None:
        login(hf_token)
    pretrained_model_name_or_path = hf_download(
        repo_id=pretrained_model_name_or_path,
        cache_dir=cache_dir,
        ignore_patterns=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.msgpack", "*.h5"],
    )
    return pretrained_model_name_or_path


def hf_download(
    repo_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
):
    # Setup cache and local dir
    local_dir = None
    if cache_dir is not None:
        cache_dir = f"{cache_dir}"
        local_dir = f"{cache_dir}/{repo_id}"

    os.makedirs(f"{cache_dir}/{repo_id}", exist_ok=True)
    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        try:
            model_path = snapshot_download(
                repo_id,
                cache_dir=cache_dir,
                local_dir=local_dir,
                local_dir_use_symlinks=True,
                revision="main",
                resume_download=True,
                token=hf_token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            break
        except requests.ReadTimeout as e:
            print(f"Read timeout: {e}")
            retry_count += 1

        except HTTPError as e:
            retry_count = max_retries
            if e.response.status_code == 401:
                print("You need to pass a valid `--hf_token=...` to download private checkpoints.")
            else:
                raise e

    return model_path


def qpc_exists(model_name: str, qpc_base_dir_name: str) -> Tuple[bool, str]:
    """
    Checks if qpc dir exists.
    Returns
    1. Boolean variable indicating if qpc files exist
    2. Path of the qpc dir if found.
    ---------
    :param model_name: str. HF Model card name.
    :param dir_path: str. Path of qpc directory.
    :return: Union[Tuple[bool, str]]: qpc_exists and path to qpc directory
    """
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
    os.makedirs(model_card_dir, exist_ok=True)

    qpc_dir_path = os.path.join(model_card_dir, qpc_base_dir_name, "qpcs")

    # Compute the boolean indicating if the QPC exists
    qpc_exists_bool = os.path.isdir(qpc_dir_path) and os.path.isfile(os.path.join(qpc_dir_path, "programqpc.bin"))

    return qpc_exists_bool, qpc_dir_path


def onnx_exists(model_name: str) -> Tuple[bool, str, str]:
    """
    Checks if qpc files already exists, removes the directory if files have been manipulated.
    ---------
    :param model_name: str. HF Model card name.
    :return: Union[Tuple[bool, str, str]]: onnx_exists and path to onnx file and directory
    """
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
    os.makedirs(model_card_dir, exist_ok=True)

    onnx_dir_path = os.path.join(model_card_dir, "onnx")
    onnx_model_path = os.path.join(onnx_dir_path, model_name.replace("/", "_") + "_kv_clipped_fp16.onnx")

    # Compute the boolean indicating if the ONNX model exists
    onnx_exists_bool = os.path.isfile(onnx_model_path) and os.path.isfile(
        os.path.join(os.path.dirname(onnx_model_path), "custom_io_fp16.yaml")
    )

    # Return the boolean, onnx_dir_path, and onnx_model_path
    return onnx_exists_bool, onnx_dir_path, onnx_model_path


def load_hf_tokenizer(model_name: str, cache_dir: Optional[str] = None, hf_token: Optional[str] = None) -> Union[PreTrainedTokenizerFast, PreTrainedTokenizer]:
    if hf_token is not None:
        login(hf_token)

    # Download tokenizer along with model if it doesn't exist
    model_hf_path = hf_download(repo_id=model_name, cache_dir=cache_dir, allow_patterns=["*.json", "*.py", "*token*"])
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_cache=True, padding_side="left", trust_remote_code=True)
    return tokenizer
