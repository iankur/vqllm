import re
from typing import Dict

import torchtune
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


torchtune.models.convert_weights._FROM_HF = _FROM_HF
torchtune.models.convert_weights._FROM_META = _FROM_META
torchtune.models.convert_weights.get_mapped_key = get_mapped_key
FullModelHFCheckpointer = torchtune.utils._checkpointing.FullModelHFCheckpointer
FullModelMetaCheckpointer = torchtune.utils._checkpointing.FullModelMetaCheckpointer
