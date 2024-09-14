import gc
import os
import re
from pathlib import Path
from typing import Any, Dict

import torch
import torchtune
from torchtune import utils
from torchtune.models import convert_weights
from torchtune.utils._checkpointing._checkpointer_utils import (
    ModelType,
    safe_torch_load,
)
from torchtune.utils.logging import get_logger

logger = get_logger("DEBUG")


_FROM_HF = convert_weights._FROM_HF
_FROM_HF.update(
    {
        # vq_attn_key
        "model.layers.{}.attention.quantizer.key.{}.codebook.weight": "layers.{}.attn.quantizer.key.{}.codebook.weight",
        "model.layers.{}.attention.quantizer.key.{}.codebook.cluster_size": "layers.{}.attn.quantizer.key.{}.codebook.cluster_size",
        "model.layers.{}.attention.quantizer.key.{}.codebook.embed_avg": "layers.{}.attn.quantizer.key.{}.codebook.embed_avg",
        "model.layers.{}.attention.quantizer.key.{}.data_initialized": "layers.{}.attn.quantizer.key.{}.data_initialized",
        # vq_attn_value
        "model.layers.{}.attention.quantizer.value.{}.codebook.weight": "layers.{}.attn.quantizer.value.{}.codebook.weight",
        "model.layers.{}.attention.quantizer.value.{}.codebook.cluster_size": "layers.{}.attn.quantizer.value.{}.codebook.cluster_size",  # noqa: B950
        "model.layers.{}.attention.quantizer.value.{}.codebook.embed_avg": "layers.{}.attn.quantizer.value.{}.codebook.embed_avg",
        "model.layers.{}.attention.quantizer.value.{}.data_initialized": "layers.{}.attn.quantizer.value.{}.data_initialized",
    }
)

_FROM_META = convert_weights._FROM_META
_FROM_META.update(
    {
        # vq_attn_key
        "layers.{}.attention.quantizer.key.{}.codebook.weight": "layers.{}.attn.quantizer.key.{}.codebook.weight",
        "layers.{}.attention.quantizer.key.{}.codebook.cluster_size": "layers.{}.attn.quantizer.key.{}.codebook.cluster_size",
        "layers.{}.attention.quantizer.key.{}.codebook.embed_avg": "layers.{}.attn.quantizer.key.{}.codebook.embed_avg",
        "layers.{}.attention.quantizer.key.{}.data_initialized": "layers.{}.attn.quantizer.key.{}.data_initialized",
        # vq_attn_value
        "layers.{}.attention.quantizer.value.{}.codebook.weight": "layers.{}.attn.quantizer.value.{}.codebook.weight",
        "layers.{}.attention.quantizer.value.{}.codebook.cluster_size": "layers.{}.attn.quantizer.value.{}.codebook.cluster_size",
        "layers.{}.attention.quantizer.value.{}.codebook.embed_avg": "layers.{}.attn.quantizer.value.{}.codebook.embed_avg",
        "layers.{}.attention.quantizer.value.{}.data_initialized": "layers.{}.attn.quantizer.value.{}.data_initialized",
    }
)


def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
    """
    Adds support to map nested module list.
    """
    try:
        if "layers" in key:
            # Replace module index with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            module_nums = re.findall(r"\d+", key)
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(*module_nums)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key


def gemma_hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 8,
    num_kv_heads: int = 1,
    dim: int = 2048,
    head_dim: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to TorchTune's format, which contains the weights
    of a Gemma model.
    State dicts from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.
    The logic is identical to :func:`~torchtune.models.convert_weights.hf_to_tune`, but doesn't load
    output projection weights.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in HF's format.
        num_heads (int): Number of heads in the model. Defaults to 8.
        num_kv_heads (int): Number of heads in the key/value projection layers. Defaults to 1.
        dim (int): Dimension of the model. Defaults to 2048.
        head_dim (int): Dimension of the attention head. This value is explicit in Gemma confs. Defaults to 256.

    Returns:
        Dict[str, torch.Tensor]: State dict in TorchTune's format.
    """
    converted_state_dict = {}

    def _permute(t, n_heads):
        return (
            t.view(n_heads, 2, head_dim // 2, dim)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), dim)
        )

    for key, value in state_dict.items():
        if (
            "rotary_emb.inv_freq" not in key and "lm_head.weight" not in key
        ):  # Skip loading the position embeddings and output projection weights
            new_key = get_mapped_key(key, _FROM_HF)
            if "q_proj" in key:
                value = _permute(value, num_heads)
            elif "k_proj" in key:
                value = _permute(value, num_kv_heads)
            converted_state_dict[new_key] = value
    return converted_state_dict


def gemma_tune_to_hf(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 8,
    num_kv_heads: int = 1,
    dim: int = 2048,
    head_dim: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from TorchTune's format to Hugging Face's format for Gemma.

    This function takes a state dictionary in TorchTune's format, which contains the weights of a Gemma model,
    and converts it into a format that can be loaded into a Hugging Face model.
    The logic is identical to :func:`~torchtune.models.convert_weights.tune_to_hf`, but saves the tied
    output projection weights.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in TorchTune's format.
        num_heads (int, optional): Number of heads in the model. Defaults to 8.
        num_kv_heads (int, optional): Number of heads in the key/value projection layers. Defaults to 1.
        dim (int, optional): Dimension of the model. Defaults to 2048.
        head_dim (int): Dimension of the attention head. This value is explicit in Gemma confs. Defaults to 256.

    Returns:
        Dict[str, torch.Tensor]: State dict in Hugging Face's format.

    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}

    def _permute(t, n_heads):
        return (
            t.view(n_heads, head_dim // 2, 2, dim)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), dim)
        )

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        if "q_proj" in key:
            value = _permute(value, num_heads)
        elif "k_proj" in key:
            value = _permute(value, num_kv_heads)
        elif "tok_embeddings" in key:
            # HF also uses tied weights, see
            # https://github.com/huggingface/transformers/blob/14ff5dd962c1bd0a4e3adaac347ba396d8df5add/src/transformers/models/gemma/convert_gemma_weights_to_hf.py#L104
            converted_state_dict["lm_head.weight"] = value
        converted_state_dict[new_key] = value
    return converted_state_dict


def load_hf_checkpoint(self) -> Dict[str, Any]:
    self._weight_map = {}

    # merged state_dict contains keys and weights from all the checkpoint files
    merged_state_dict: Dict[str, torch.Tensor] = {}

    # converted_state_dict is the final state_dict passed to the recipe after the
    # keys are converted into the torchtune format. This optionally also contains
    # the recipe state and adapter weights
    converted_state_dict: Dict[str, Dict[str, torch.Tensor]] = {}

    for cpt_idx, cpt_path in enumerate(self._checkpoint_paths):
        state_dict = safe_torch_load(cpt_path)
        for key, value in state_dict.items():
            # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
            # will break recipe code
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Expected all values in the state dict to be torch.Tensor. "
                    f"Found {type(value)} instead."
                )
            # idx is written in the 4 digit format (eg: 0001, 0002, etc.)
            self._weight_map[key] = f"{cpt_idx+1:04}"
        merged_state_dict.update(state_dict)

        # delete the state_dict to free up memory; TODO check if this del is needed
        del state_dict
        gc.collect()

    if self._model_type == ModelType.GEMMA:
        converted_state_dict[utils.MODEL_KEY] = gemma_hf_to_tune(
            merged_state_dict,
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
            head_dim=self._config["head_dim"],
        )
    else:
        converted_state_dict[utils.MODEL_KEY] = convert_weights.hf_to_tune(
            merged_state_dict,
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
            head_dim=self._config.get("head_dim", None),
        )

    if self._resume_from_checkpoint:
        recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
        converted_state_dict.update(recipe_state)
    return converted_state_dict


def save_hf_checkpoint(
    self,
    state_dict: Dict[str, Any],
    epoch: int,
    intermediate_checkpoint: bool = False,
) -> None:
    if self._model_type == ModelType.GEMMA:
        state_dict[utils.MODEL_KEY] = gemma_tune_to_hf(
            state_dict[utils.MODEL_KEY],
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
            head_dim=self._config["head_dim"],
        )
    else:
        state_dict[utils.MODEL_KEY] = convert_weights.tune_to_hf(
            state_dict[utils.MODEL_KEY],
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
            head_dim=self._config["head_dim"],
        )
    self._output_dir.mkdir(exist_ok=True)

    # split the state_dict into separate dicts, one for each output checkpoint file
    split_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
    MAX_CPT_IDX = max(map(int, self._weight_map.values()))  # noqa: N806
    MAX_CPT_IDX = f"{MAX_CPT_IDX+1:04}"  # noqa: N806
    for key, weight in state_dict[utils.MODEL_KEY].items():
        cpt_idx = self._weight_map[key] if key in self._weight_map else MAX_CPT_IDX
        if cpt_idx not in split_state_dicts:
            split_state_dicts[cpt_idx] = {}
        split_state_dicts[cpt_idx].update({key: weight})

    # write the partitioned state dicts to the right checkpoint file
    for cpt_idx, model_state_dict in split_state_dicts.items():
        output_path = Path.joinpath(
            self._output_dir, f"hf_model_{cpt_idx}_{epoch}"
        ).with_suffix(".pt")
        torch.save(model_state_dict, output_path)
        logger.info(
            "Model checkpoint of size "
            f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
            f"saved to {output_path}"
        )


torchtune.models.convert_weights._FROM_HF = _FROM_HF
torchtune.models.convert_weights._FROM_META = _FROM_META
torchtune.models.convert_weights.get_mapped_key = get_mapped_key
FullModelHFCheckpointer = torchtune.utils._checkpointing.FullModelHFCheckpointer
FullModelHFCheckpointer.load_checkpoint = load_hf_checkpoint
FullModelHFCheckpointer.save_checkpoint = save_hf_checkpoint
FullModelMetaCheckpointer = torchtune.utils._checkpointing.FullModelMetaCheckpointer
