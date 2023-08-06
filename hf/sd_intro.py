import torch
from diffusers import StableDiffusionPipeline

from hf.common import figure_out_device


def diy_sample(device, pipe, generator):
    guidance_scale = 8  # @param
    num_inference_steps = 30  # @param
    prompt = "Beautiful picture of a wave breaking"  # @param
    negative_prompt = "zoomed in, blurry, oversaturated, warped"  # @param

    # Encode the prompt
    text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)

    # Create our random starting point
    latents = torch.randn((1, 4, 64, 64), device=device, generator=generator)
    latents *= pipe.scheduler.init_noise_sigma

    # Prepare the scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Loop through the sampling timesteps
    for i, t in enumerate(pipe.scheduler.timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)

        # Apply any scaling required by the scheduler
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual with the unet
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode the resulting latents into an image
    with torch.no_grad():
        image = pipe.decode_latents(latents.detach())

    return image


def main():
    device = figure_out_device()
    generator = torch.Generator(device=device).manual_seed(42)
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    image = diy_sample(device, pipe, generator)
    # View
    # pipe.numpy_to_pil(image)[0]
