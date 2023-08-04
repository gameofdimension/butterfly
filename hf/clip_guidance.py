import open_clip
import torch

import torchvision
from tqdm import tqdm


def get_clip_loss_func(device):
    # @markdown load a CLIP model and define the loss function

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model.to(device)

    # Transforms to resize and augment an image + normalize to match CLIP's training data
    tfms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(224),  # Random CROP each time
            torchvision.transforms.RandomAffine(
                5
            ),  # One possible random augmentation: skews the image
            torchvision.transforms.RandomHorizontalFlip(),  # You can add additional augmentations if you like
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    # And define a loss function that takes an image, embeds it and compares with
    # the text features of the prompt
    def clip_loss(image, text_features):
        image_features = clip_model.encode_image(
            tfms(image)
        )  # Note: applies the above transforms
        input_normed = torch.nn.functional.normalize(image_features.unsqueeze(1), dim=2)
        embed_normed = torch.nn.functional.normalize(text_features.unsqueeze(0), dim=2)
        dists = (
            input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        )  # Squared Great Circle Distance
        return dists.mean()

    return clip_model, clip_loss


def guidance(prompt, scheduler, device, clip_model, clip_loss, image_pipe):
    # @markdown applying guidance using CLIP

    # prompt = "Red Rose (still life), red flower painting"  # @param

    # Explore changing this
    guidance_scale = 8  # @param
    n_cuts = 4  # @param

    # More steps -> more time for the guidance to have an effect
    scheduler.set_timesteps(50)

    # We embed a prompt with CLIP as our target
    text = open_clip.tokenize([prompt]).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_model.encode_text(text)

    x = torch.randn(4, 3, 256, 256).to(
        device
    )  # RAM usage is high, you may want only 1 image at a time

    for i, t in tqdm(enumerate(scheduler.timesteps)):

        model_input = scheduler.scale_model_input(x, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = image_pipe.unet(model_input, t)["sample"]

        cond_grad = 0
        loss = 0

        for cut in range(n_cuts):
            # Set requires grad on x
            x = x.detach().requires_grad_()

            # Get the predicted x0:
            x0 = scheduler.step(noise_pred, t, x).pred_original_sample

            # Calculate loss
            loss = clip_loss(x0, text_features) * guidance_scale

            # Get gradient (scale by n_cuts since we want the average)
            cond_grad -= torch.autograd.grad(loss, x)[0] / n_cuts

        if i % 25 == 0:
            print("Step:", i, ", Guidance loss:", loss.item())

        # Modify x based on this gradient
        alpha_bar = scheduler.alphas_cumprod[i]
        x = (
                x.detach() + cond_grad * alpha_bar.sqrt()
        )  # Note the additional scaling factor here!

        # Now step with scheduler
        x = scheduler.step(noise_pred, t, x).prev_sample

    # grid = torchvision.utils.make_grid(x.detach(), nrow=4)
    # im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
    # Image.fromarray(np.array(im * 255).astype(np.uint8))
    return x.detach()
