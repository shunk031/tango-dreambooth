import logging
from typing import Optional

import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from tango import Step
from tango.format import Format

from dreambooth.integrations.diffusers import DiffusersPipelineFormat

logger = logging.getLogger(__name__)


@Step.register("create_pipeline")
class CreatePipeline(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = False

    FORMAT: Format = DiffusersPipelineFormat()

    def run(  # type: ignore
        self,
        model_name: str,
        model: nn.Module,
    ) -> StableDiffusionPipeline:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            unet=model.unet,
            text_encoder=model.text_encoder,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(
                model_name, subfolder="scheduler"
            ),
        )

        return pipeline
