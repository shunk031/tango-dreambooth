import logging
import os
from typing import Optional

import datasets as ds
import torch
from diffusers import StableDiffusionPipeline
from tango import Step
from tango.integrations.torch.util import resolve_device
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@Step.register("generate_class_image")
class GenerateClassImage(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True

    def generate_class_image(
        self,
        class_images_dir: str,
        num_class_images: int,
        model_name: str,
        sample_batch_size: int,
        prior_preservation_class_prompt: str,
        revision: str = "fp16",
    ) -> None:
        if not os.path.exists(class_images_dir):
            logger.info(f"Make directory to {class_images_dir}")
            os.makedirs(class_images_dir)

        cur_class_images = len(os.listdir(class_images_dir))

        if cur_class_images >= num_class_images:
            return None

        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.float16,
        )
        pipe = pipe.to(resolve_device())

        pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)

        num_new_images = num_class_images - cur_class_images
        logger.info(f"Number of class images to sample: {num_new_images}")

        it = range(0, num_new_images, sample_batch_size)
        for idx in tqdm(it, desc="Generating class images"):
            images = pipe(
                prompt=prior_preservation_class_prompt,
                num_images_per_prompt=sample_batch_size,
            ).images
            for i, image in enumerate(images):
                image_name = f"{cur_class_images + idx + i}.jpg"
                image_save_path = os.path.join(class_images_dir, image_name)
                image.save(image_save_path)

    def run(  # type: ignore[override]
        self,
        is_prior_preservation: bool,
        model_name: Optional[str] = None,
        class_images_dir: Optional[str] = None,
        num_class_images: Optional[int] = None,
        sample_batch_size: Optional[int] = None,
        prior_preservation_class_prompt: Optional[str] = None,
    ) -> Optional[ds.DatasetDict]:
        if not is_prior_preservation:
            return None

        if (
            class_images_dir is not None
            and model_name
            and num_class_images is not None
            and sample_batch_size is not None
            and prior_preservation_class_prompt is not None
        ):
            self.generate_class_image(
                class_images_dir=class_images_dir,
                num_class_images=num_class_images,
                prior_preservation_class_prompt=prior_preservation_class_prompt,
                model_name=model_name,
                sample_batch_size=sample_batch_size,
            )
            return ds.load_dataset("imagefolder", data_dir=class_images_dir)

        else:
            raise ValueError(
                "Even though `is_prior_preservation` is True, the following values may not be set:\n"
                f"- `class_images_dir` (= {class_images_dir})\n"
                f"- `model_name (= {model_name})`\n"
                f"- `num_images_dir` (= {num_class_images})\n"
                f"- `sample_batch_size` (= {sample_batch_size})\n"
                f"- `prior_preservation_class_prompt` (= {prior_preservation_class_prompt})\n"
            )
