from typing import Dict

import torch as th
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from tango.integrations.torch.model import Model
from transformers import CLIPTextModel


@Model.register("stable_diffusion")
class StableDiffusionModel(Model):
    def __init__(
        self,
        model_name: str,
        is_prior_preservation: bool = False,
        is_train_text_encoder: bool = False,
        is_gradient_checkpointing: bool = False,
        prior_loss_weight: float = 0.5,
        scale_factor: float = 0.18215,
    ) -> None:
        super().__init__()

        self.is_prior_preservation = is_prior_preservation
        self.is_train_text_encoder = is_train_text_encoder
        self.is_gradient_checkpointing = is_gradient_checkpointing
        self.prior_loss_weight = prior_loss_weight
        self.scale_factor = scale_factor

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder="vae",
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name,
            subfolder="unet",
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_name,
            subfolder="scheduler",
        )

        self.vae.requires_grad_(False)
        if not self.is_train_text_encoder:
            self.text_encoder.requires_grad_(False)

        if is_gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.is_train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

    def forward(
        self,
        pixel_values: th.Tensor,
        input_ids: th.Tensor,
    ) -> Dict[str, th.Tensor]:
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.scale_factor

        noise = th.randn_like(latents)
        bsz = latents.size(dim=0)

        timesteps = th.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(input_ids).last_hidden_state

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        if self.is_prior_preservation:
            noise_pred, noise_pred_prior = th.chunk(noise_pred, 2, dim=0)
            target, target_prior = th.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = (
                F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                .mean([1, 2, 3])
                .mean()
            )

            # Compute prior loss
            prior_loss = F.mse_loss(
                noise_pred_prior.float(), target_prior.float(), reduction="mean"
            )

            # Add the prior loss to the instance loss.
            loss = loss + self.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        return {"loss": loss}
