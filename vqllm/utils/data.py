import random
from typing import Any, Dict, List, Mapping, Optional

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, truncate
from torchtune.datasets import (
    PackedDataset,
    TextCompletionDataset as OriginalTextCompletionDataset,
)
from torchtune.modules.tokenizers import Tokenizer


class TextCompletionDataset(OriginalTextCompletionDataset):
    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        prompt = sample[self._column]
        tokens = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=True)

        # TODO I don't understand the truncation logic here
        # There seems to be an edge case where the EOS token is removed
        # Truncate if needed, but don't coerce EOS id
        if self.max_seq_len is not None:
            tokens = truncate(tokens, self.max_seq_len - 1)

        # No need to offset labels by 1 - happens in the recipe
        # labels = tokens.copy()

        # We create the label explicitly here as packed dataset results
        # in eos token of first sequence as input and  bos token
        # of second sequence as label, when two sequences are packed
        labels = tokens[1:] + [CROSS_ENTROPY_IGNORE_IDX]

        return {"tokens": tokens, "labels": labels}


def text_completion_dataset(
    tokenizer: Tokenizer,
    source: str = None,
    column: str = None,
    max_seq_len: Optional[int] = None,
    num_random_samples: Optional[int] = None,
    packed: Optional[bool] = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> TextCompletionDataset:
    """
    Samples a random subset from any HF dataset supported by torchtune text_completion_dataset.
    This is convinient to avoid packing of the entire data when packed is set to True.
    """
    ds = TextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        column=column,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )

    if num_random_samples:
        if num_random_samples > len(ds._data):
            raise ValueError(
                f"num_random_samples={num_random_samples} is greater than the length of the dataset={len(ds._data)}"
            )
        ds._data = ds._data.select(
            random.sample(range(len(ds._data)), num_random_samples)
        )
    return (
        PackedDataset(ds, max_seq_len=max_seq_len, padding_idx=tokenizer.pad_id)
        if packed
        else ds
    )
