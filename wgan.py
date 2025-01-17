import torch
from torch import nn, optim, autograd
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from dataclasses import dataclass
import time
torch.set_num_threads(1)
torch.manual_seed(1)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

@dataclass
class Hyperparameter:
    batchsize: int          = 64
    num_epochs: int         = 5
    latent_size: int        = 32
    n_critic: int           = 5
    critic_size: int        = 1024
    generator_size: int     = 1024
    critic_hidden_size: int = 1024
    gp_lambda: float        = 10.

hp = Hyperparameter()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

dataset  = torchvision.datasets.MNIST("mnist", download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=hp.batchsize, num_workers=1, shuffle=True, drop_last=True, pin_memory=True)



# Define the generator

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Add latent embedding layer to adjust the dimension of the input
        self.latent_embedding = nn.Linear(hp.latent_size, hp.generator_size * 1 * 1)

        # Transposed CNN layers to transfer noise to image
        self.tcnn = nn.Sequential(
        # input is Z, going into a convolution
        nn.ConvTranspose2d(hp.generator_size, hp.generator_size, 4, 1, 0),
        nn.BatchNorm2d(hp.generator_size),
        nn.ReLU(inplace=True),
        # upscaling
        nn.ConvTranspose2d(hp.generator_size, hp.generator_size // 2, 3, 2, 1),
        nn.BatchNorm2d(hp.generator_size // 2),
        nn.ReLU(inplace=True),
        # upscaling
        nn.ConvTranspose2d(hp.generator_size // 2, hp.generator_size // 4, 4, 2, 1),
        nn.BatchNorm2d(hp.generator_size // 4),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(hp.generator_size // 4, 1, 4, 2, 1),
        nn.Tanh()
        )


    def forward(self, latent):
        vec_latent = self.latent_embedding(latent).reshape(-1, hp.generator_size, 1, 1)
        return self.tcnn(vec_latent)


# Define the critic

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        # CNN layers that perform downscaling
        self.cnn_net = nn.Sequential(
        nn.Conv2d(1, hp.critic_size // 4, 3, 2),
        nn.InstanceNorm2d(hp.critic_size // 4, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hp.critic_size // 4, hp.critic_size // 2, 3, 2),
        nn.InstanceNorm2d(hp.critic_size // 2, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hp.critic_size // 2, hp.critic_size, 3, 2),
        nn.InstanceNorm2d(hp.critic_size, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Flatten(),
        )

        # Linear layers that produce the output from the features
        self.critic_net = nn.Sequential(
        nn.Linear(hp.critic_size * 4, hp.critic_hidden_size),
        nn.LeakyReLU(0.2, inplace=True),

        # Add the last layer to reflect the output
        nn.Linear(hp.critic_hidden_size, 1)
        )

    def forward(self, image):
        cnn_features = self.cnn_net(image)
        return self.critic_net(cnn_features)



critic, generator = Critic().to(device), Generator().to(device)

critic_optimizer = optim.AdamW(critic.parameters(), lr=1e-4,betas=(0., 0.9))
generator_optimizer = optim.AdamW(generator.parameters(), lr=1e-4,betas=(0., 0.9))

img_list, generator_losses, critic_losses = [], [], []
iters = 0
fixed_noise = torch.randn((64, hp.latent_size), device=device)
grad_tensor = torch.ones((hp.batchsize, 1), device=device)
start_time = time.time()
for epoch in range(hp.num_epochs):
    for batch_idx, data in enumerate(dataloader, 0):
        real_images = data[0].to(device)

        # Update Critic
        critic_optimizer.zero_grad()

        # (a) Real loss
        critic_output_real = critic(real_images)
        critic_loss_real = critic_output_real.mean()

        # (b) Fake loss
        # Implement the fake loss

        # (1) Generating a noise tensor (of dimension (batch_size, latent_size)), you are required to
        # use the hyperparameters in the hp class

        noise = torch.randn((hp.batchsize, hp.latent_size), device=device)

        # (2) Generate fake images using the generator (hint: you are not supposed to perform gradient
        # update on the generator)

        fake_image = generator(noise).detach()

        # (3) Calculate the fake loss using the output of the generator
        critic_output_fake =  critic(fake_image)
        critic_loss_fake = critic_output_fake.mean()

        #  (c) Gradient penalty
        alpha = torch.rand((hp.batchsize, 1, 1, 1), device=device)
        interpolates = (alpha * real_images + ((1. - alpha) * fake_image)).requires_grad_(True)
        d_interpolates = critic(interpolates)
        gradients = autograd.grad(d_interpolates, interpolates, grad_tensor, create_graph=True, only_inputs=True)[0]
        gradient_penalty = hp.gp_lambda * ((gradients.view(hp.batchsize, -1).norm(dim=1) - 1.) ** 2).mean()

        critic_loss = -critic_loss_real + critic_loss_fake + gradient_penalty

        critic_loss.backward()
        critic_optimizer.step()

        if batch_idx % hp.n_critic == 0:
            # Update Generator
            generator_optimizer.zero_grad()

            noise = torch.randn((hp.batchsize, hp.latent_size), device=device)
            fake_image = generator(noise)
            critic_output_fake = critic(fake_image)
            generator_loss = -critic_output_fake.mean()

            generator_loss.backward()
            generator_optimizer.step()

        # Output training stats
        if batch_idx % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"[{epoch:>2}/{hp.num_epochs}][{iters:>7}][{elapsed_time:8.2f}s]\t"
                  f"d_loss/g_loss: {critic_loss.item():4.2}/{generator_loss.item():4.2}\t")

        # Save Losses for plotting later
        generator_losses.append(generator_loss.item())
        critic_losses.append(critic_loss.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == hp.num_epochs - 1) and (batch_idx == len(dataloader) - 1)):
            with torch.no_grad(): fake_images = generator(fixed_noise).cpu()
            img_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))

        iters += 1