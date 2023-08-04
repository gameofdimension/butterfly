import torch
from tqdm import tqdm


def color_loss(images, target_color=(0.1, 0.9, 0.5)):
    """Given a target color (R, G, B) return a loss for how far away on average
    the images' pixels are from that color. Defaults to a light teal: (0.1, 0.9, 0.5)"""
    target = (
            torch.tensor(target_color).to(images.device) * 2 - 1
    )  # Map target color to (-1, 1)
    target = target[
             None, :, None, None
             ]  # Get shape right to work with the images (b, c, h, w)
    error = torch.abs(
        images - target
    ).mean()  # Mean absolute difference between the image pixels and the target color
    return error


def color_guidance_1(device, scheduler, image_pipe):
    # Variant 1: shortcut method

    # The guidance scale determines the strength of the effect
    guidance_loss_scale = 40  # Explore changing this to 5, or 100

    x = torch.randn(8, 3, 256, 256).to(device)

    for i, t in tqdm(enumerate(scheduler.timesteps)):

        # Prepare the model input
        model_input = scheduler.scale_model_input(x, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = image_pipe.unet(model_input, t)["sample"]

        # Set x.requires_grad to True
        x = x.detach().requires_grad_()

        # Get the predicted x0
        x0 = scheduler.step(noise_pred, t, x).pred_original_sample

        # Calculate loss
        loss = color_loss(x0) * guidance_loss_scale
        if i % 10 == 0:
            print(i, "loss:", loss.item())

        # Get gradient
        cond_grad = -torch.autograd.grad(loss, x)[0]

        # Modify x based on this gradient
        x = x.detach() + cond_grad

        # Now step with scheduler
        x = scheduler.step(noise_pred, t, x).prev_sample

    # View the output
    # grid = torchvision.utils.make_grid(x, nrow=4)
    # im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
    # Image.fromarray(np.array(im * 255).astype(np.uint8))
    return x


def color_guidance_2(device, scheduler, image_pipe):
    # Variant 2: setting x.requires_grad before calculating the model predictions

    guidance_loss_scale = 40
    x = torch.randn(4, 3, 256, 256).to(device)

    for i, t in tqdm(enumerate(scheduler.timesteps)):

        # Set requires_grad before the model forward pass
        x = x.detach().requires_grad_()
        model_input = scheduler.scale_model_input(x, t)

        # predict (with grad this time)
        noise_pred = image_pipe.unet(model_input, t)["sample"]

        # Get the predicted x0:
        x0 = scheduler.step(noise_pred, t, x).pred_original_sample

        # Calculate loss
        loss = color_loss(x0) * guidance_loss_scale
        if i % 10 == 0:
            print(i, "loss:", loss.item())

        # Get gradient
        cond_grad = -torch.autograd.grad(loss, x)[0]

        # Modify x based on this gradient
        x = x.detach() + cond_grad

        # Now step with scheduler
        x = scheduler.step(noise_pred, t, x).prev_sample

    # grid = torchvision.utils.make_grid(x, nrow=4)
    # im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
    # Image.fromarray(np.array(im * 255).astype(np.uint8))
    return x


