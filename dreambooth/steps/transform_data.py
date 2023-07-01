from typing import Optional, TypedDict

import datasets as ds
import torch as th
import torchvision.transforms as transforms
from tango import Step
from tango.integrations.transformers import Tokenizer


class PreprocessedExample(TypedDict):
    instance_images: th.Tensor
    instance_prompt_ids: th.Tensor
    class_images: Optional[th.Tensor]
    class_prompt_ids: Optional[th.Tensor]


@Step.register("transform_data")
class TransformData(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = False

    def raw_example_transformer(
        self,
        idx: int,
        instance_example,
        instance_prompt: str,
        tokenizer: Tokenizer,
        image_transforms: transforms.Compose,
        class_dataset: Optional[ds.DatasetDict] = None,
        class_prompt: Optional[str] = None,
    ) -> PreprocessedExample:
        instance_image_pl = instance_example["image"]
        if not instance_image_pl.mode == "RGB":
            instance_image_pl = instance_image_pl.convert("RGB")
        instance_image_th = image_transforms(instance_image_pl)

        instance_prompt_ids = tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids
        example: PreprocessedExample = {
            "instance_images": instance_image_th,
            "instance_prompt_ids": instance_prompt_ids,
            "class_images": None,
            "class_prompt_ids": None,
        }

        if class_dataset is not None and class_prompt is not None:
            num_class_images = class_dataset["train"].num_rows
            class_image_pl = class_dataset["train"][idx % num_class_images]["image"]
            if class_image_pl.mode == "RGB":
                class_image_pl = class_image_pl.convert("RGB")
            class_image_th = image_transforms(class_image_pl)
            example["class_images"] = class_image_th

            class_prompt_ids = tokenizer(
                class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).input_ids
            example["class_prompt_ids"] = class_prompt_ids

        return example

    def run(  # type: ignore[override]
        self,
        instance_dataset: ds.DatasetDict,
        tokenizer: Tokenizer,
        instance_prompt: str,
        image_size: int = 512,
        is_center_crop: bool = False,
        class_prompt: Optional[str] = None,
        class_dataset: Optional[ds.DatasetDict] = None,
    ) -> ds.DatasetDict:
        image_transforms = transforms.Compose(
            (
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(image_size)
                if is_center_crop
                else transforms.RandomCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            )
        )

        def raw_example_transformer_wrapper(instance_example, idx: int):
            return self.raw_example_transformer(
                idx=idx,
                instance_example=instance_example,
                instance_prompt=instance_prompt,
                tokenizer=tokenizer,
                class_dataset=class_dataset,
                class_prompt=class_prompt,
                image_transforms=image_transforms,
            )

        dataset = instance_dataset.map(
            raw_example_transformer_wrapper,
            with_indices=True,
            remove_columns=["image"],
        )
        dataset = dataset.with_format(type="torch")

        return dataset
