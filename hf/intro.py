import torch
from datasets import load_dataset
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from torchvision import transforms


def get_data_loader(image_size: int, batch_size: int = 64):
    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

    # Or load images from a local folder
    # dataset = load_dataset("imagefolder", data_dir="path/to/folder")

    # We'll train on 32-pixel square images, but you can try larger sizes too
    # image_size = 32
    # You can lower your batch size if you're running out of GPU memory
    # batch_size = 64

    # Define data augmentations
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize
            transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
            transforms.ToTensor(),  # Convert to tensor (0, 1)
            transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    # Create a dataloader from the dataset to serve up the transformed images in batches
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return train_dataloader


def create_model(image_size, device):
    # Create a model
    model = UNet2DModel(
        sample_size=image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
    )
    model.to(device)
    return model


def train(model, train_dataloader, device, scheduler_step: int, epoch: int):
    # Set the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=scheduler_step, beta_schedule="squaredcos_cap_v2"
    )

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

    losses = []

    for epoch in range(epoch):
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate the loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward(loss)
            losses.append(loss.item())

            # Update the model parameters with the optimizer
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 5 == 0:
            loss_last_epoch = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
            print(f"Epoch:{epoch + 1}, loss: {loss_last_epoch}")
    return noise_scheduler, model


def save_as_pipeline(noise_scheduler, model, save_path):
    image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    image_pipe.save_pretrained(save_path)


def sample_images(n: int, device, noise_scheduler, model):
    sample = torch.randn(n, 3, 32, 32).to(device)

    for i, t in enumerate(noise_scheduler.timesteps):
        # Get model pred
        with torch.no_grad():
            residual = model(sample, t).sample

        # Update sample with step
        sample = noise_scheduler.step(residual, t, sample).prev_sample
    return sample
