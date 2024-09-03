import os
import re
from pathlib import Path
from typing import Any, Dict

import torch
import torchtune
from torchtune import utils
from torchtune.models import convert_weights
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


def save_hf_checkpoint(
    self,
    state_dict: Dict[str, Any],
    epoch: int,
    intermediate_checkpoint: bool = False,
) -> None:
    state_dict[utils.MODEL_KEY] = convert_weights.tune_to_hf(
        state_dict[utils.MODEL_KEY],
        num_heads=self._config["num_attention_heads"],
        num_kv_heads=self._config["num_key_value_heads"],
        dim=self._config["hidden_size"],
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
FullModelHFCheckpointer.save_checkpoint = save_hf_checkpoint
FullModelMetaCheckpointer = torchtune.utils._checkpointing.FullModelMetaCheckpointer
