import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader


def get_tain_dataloader(batch_size: int):
    dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True,
                                         transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount


class BasicUNet(nn.Module):
    """A minimal UNet implementation."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU()  # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x


def train(n_epochs, device):
    # Dataloader (you can mess with batch size)
    batch_size = 128
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = get_tain_dataloader(batch_size=batch_size)

    # How many runs through the data should we do?
    # n_epochs = 3

    # Create the network
    net = BasicUNet()
    net.to(device)

    # Our loss function
    loss_fn = nn.MSELoss()

    # The optimizer
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Keeping a record of the losses for later viewing
    losses = []

    # The training loop
    for epoch in range(n_epochs):

        for x, y in train_dataloader:
            # Get some data and prepare the corrupted version
            x = x.to(device)  # Data on the GPU
            noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts
            noisy_x = corrupt(x, noise_amount)  # Create our noisy x

            # Get the model prediction
            pred = net(noisy_x)

            # Calculate the loss
            loss = loss_fn(pred, x)  # How close is the output to the true 'clean' x?

            # Backprop and update the params:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Store the loss for later
            losses.append(loss.item())

        # Print our the average of the loss values for this epoch:
        avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
        print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')
    return net


def sample_images(n_steps: int, device, net: BasicUNet):
    # n_steps = 5
    x = torch.rand(8, 1, 28, 28).to(device)  # Start from random
    step_history = [x.detach().cpu()]
    pred_output_history = []

    for i in range(n_steps):
        with torch.no_grad():  # No need to track gradients during inference
            pred = net(x)  # Predict the denoised x0
        pred_output_history.append(pred.detach().cpu())  # Store model output for plotting
        mix_factor = 1 / (n_steps - i)  # How much we move towards the prediction
        x = x * (1 - mix_factor) + pred * mix_factor  # Move part of the way there
        step_history.append(x.detach().cpu())  # Store step for plotting

    # fig, axs = plt.subplots(n_steps, 2, figsize=(9, 4), sharex=True)
    # axs[0, 0].set_title('x (model input)')
    # axs[0, 1].set_title('model prediction')
    # for i in range(n_steps):
    #     axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap='Greys')
    #     axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap='Greys')
    return pred_output_history, step_history
