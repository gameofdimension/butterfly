import math
from argparse import Namespace

import torch
import torch.nn.functional as F
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


class DreamBoothDataset(Dataset):
    def __init__(self, dataset, instance_prompt, tokenizer, size=512):
        self.dataset = dataset
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        image = self.dataset[index]["image"]
        example["instance_images"] = self.transforms(image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example


def build_dataset(model_id, instance_prompt, img_dir):
    # The Stable Diffusion checkpoint we'll fine-tune
    # model_id = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
    )

    dataset = load_dataset("imagefolder", data_dir=img_dir, split='train')
    # Remove the dummy label column
    # dataset = dataset.remove_columns("label")

    # Push to Hub
    # dataset.push_to_hub("dreambooth-hackathon-images")

    train_dataset = DreamBoothDataset(dataset, instance_prompt, tokenizer)
    return train_dataset


def collate_fn(tokenizer, examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    ).input_ids

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


def get_instance_prompt(name, classname):
    name_of_your_concept = name  # "ccorgi"  # CHANGE THIS ACCORDING TO YOUR SUBJECT
    type_of_thing = classname  # "dog"  # CHANGE THIS ACCORDING TO YOUR SUBJECT
    instance_prompt = f"a photo of {name_of_your_concept} {type_of_thing}"
    return instance_prompt


def build_args(model_id, train_dataset, instance_prompt):
    learning_rate = 2e-06
    max_train_steps = 400

    args = Namespace(
        pretrained_model_name_or_path=model_id,
        resolution=512,  # Reduce this if you want to save some memory
        train_dataset=train_dataset,
        instance_prompt=instance_prompt,
        learning_rate=learning_rate,
        max_train_steps=max_train_steps,
        train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase this if you want to lower memory usage
        max_grad_norm=1.0,
        gradient_checkpointing=True,  # set this to True to lower the memory usage.
        use_8bit_adam=True,  # use 8bit optimizer from bitsandbytes
        seed=3434554,
        sample_batch_size=2,
        output_dir="my-dreambooth",  # where to save the pipeline
    )
    return args


def prepapre_model(model_id):
    # model_id = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
    return text_encoder, vae, unet, feature_extractor, tokenizer


def training_function(text_encoder, vae, unet, feature_extractor, tokenizer, args: Namespace):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    set_seed(args.seed)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        unet.parameters(),  # only optimize unet
        lr=args.learning_rate,
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    train_dataloader = DataLoader(
        args.train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(tokenizer, x),
    )

    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # Move text_encode and vae to gpu
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
    )
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.get("num_train_timesteps"),
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                        .mean([1, 2, 3])
                        .mean()
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        print(f"Loading pipeline and saving to {args.output_dir}...")
        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            steps_offset=1,
        )
        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ),
            feature_extractor=feature_extractor,
        )
        pipeline.save_pretrained(args.output_dir)


def main():
    model_id = "CompVis/stable-diffusion-v1-4"
    instance_prompt = get_instance_prompt(name='ccorgi', classname='dog')
    img_dir = '/Users/yzq/Downloads/corgi'
    train_dataset = build_dataset(model_id=model_id, instance_prompt=instance_prompt, img_dir=img_dir)
    args = build_args(model_id=model_id, train_dataset=train_dataset, instance_prompt=instance_prompt)

    num_of_gpus = 1  # CHANGE THIS TO MATCH THE NUMBER OF GPUS YOU HAVE
    text_encoder, vae, unet, feature_extractor, tokenizer = prepapre_model(model_id=model_id)
    notebook_launcher(
        training_function,
        args=(text_encoder, vae, unet, feature_extractor, tokenizer, args),
        num_processes=num_of_gpus
    )
