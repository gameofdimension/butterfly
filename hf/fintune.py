import torch
import torchvision
from datasets import load_dataset
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm


def build_dataloader(batch_size: int):
    # @markdown load and prepare a dataset:
    # Not on Colab? Comments with #@ enable UI tweaks like headings or user inputs
    # but can safely be ignored if you're working on a different platform.

    dataset_name = "huggan/smithsonian_butterflies_subset"  # @param
    dataset = load_dataset(dataset_name, split="train")
    image_size = 256  # @param
    # batch_size = 4  # @param
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # print("Previewing batch:")
    # batch = next(iter(train_dataloader))
    # grid = torchvision.utils.make_grid(batch["images"], nrow=4)
    # plt.imshow(grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5);
    return train_dataloader


def train(device, image_pipe, train_dataloader):
    num_epochs = 2  # @param
    lr = 1e-5  # 2param
    grad_accumulation_steps = 2  # @param

    optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=lr)

    losses = []

    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            clean_images = batch["images"].to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                image_pipe.scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction for the noise
            noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

            # Compare the prediction with the actual noise:
            loss = torch.nn.functional.mse_loss(
                noise_pred, noise
            )  # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)

            # Store for later plotting
            losses.append(loss.item())

            # Update the model parameters with the optimizer based on this loss
            loss.backward(loss)

            # Gradient accumulation:
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        print(
            f"Epoch {epoch} average loss: {sum(losses[-len(train_dataloader):]) / len(train_dataloader)}"
        )

    # Plot the loss curve:
    # plt.plot(losses)
    return image_pipe


def sample_images(device, scheduler, image_pipe):
    x = torch.randn(8, 3, 256, 256).to(device)  # Batch of 8
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        model_input = scheduler.scale_model_input(x, t)
        with torch.no_grad():
            noise_pred = image_pipe.unet(model_input, t)["sample"]
        x = scheduler.step(noise_pred, t, x).prev_sample
    grid = torchvision.utils.make_grid(x, nrow=4)
    plt.imshow(grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)
