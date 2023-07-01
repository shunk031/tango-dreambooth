from typing import Any, Dict, List, TypedDict

import torch as th
from tango.integrations.torch import DataCollator
from tango.integrations.transformers import Tokenizer

from dreambooth.steps.transform_data import PreprocessedExample


class BatchExample(TypedDict):
    input_ids: th.Tensor
    pixel_values: th.Tensor


@DataCollator.register("custom_collator")
class CustomCollator(DataCollator[PreprocessedExample]):
    def __init__(self, tokenizer: Tokenizer, is_prior_preservation: bool) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.is_prior_preservation = is_prior_preservation

    def __call__(self, items: List[PreprocessedExample]) -> BatchExample:
        input_ids = [item["instance_prompt_ids"] for item in items]
        pixel_values_list = [item["instance_images"] for item in items]

        if self.is_prior_preservation:
            input_ids += [item["class_prompt_ids"] for item in items]  # type: ignore
            pixel_values_list += [item["class_images"] for item in items]  # type: ignore

        pixel_values = th.stack(pixel_values_list)
        pixel_values = pixel_values.to(memory_format=th.contiguous_format).float()

        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        batch: BatchExample = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch
