import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-4
z_dim = 64
image_dim = 784
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn(batch_size, z_dim, device=device)
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

run_count = 2
dataset = datasets.MNIST(root="dataset/", train=True, download=True, transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST{str(run_count)}/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST{str(run_count)}/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, image_dim).to(device)
        batch_size = real.shape[0]

        # Train Discriminator:  maximize log(D(real)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim, device=device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))  # log(D(real))
        disc_fake = disc(fake).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  # log(1 - D(G(z)))  
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()


        # Train Generator: maximize log(D(G(z))) <---> minimize log(1 - D(G(z)))
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()


        if batch_idx == 0:
            print( f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Mnist Real Images", img_grid_real, global_step=step)
                step += 1


# writer_fake.close()
# writer_real.close()

# Things to try:
# 1. Larger network
# 2. Better normalization with BatchNorm
# 3. Different learning rate
# 4. Change to a CNN
