import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from config import Config


def train_real_batch(data, netD, criterion, optimizerD, real_label, device):
    netD.zero_grad()
    real_cpu = data["image"].to(device)
    b_size = Config.batch_size
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
    output = netD(real_cpu).view(-1)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()
    return errD_real, D_x


def train_fake_batch(netG, netD, criterion, optimizerD, fake_label, nz, device):
    noise = torch.randn(Config.batch_size, nz, 1, 1, device=device)
    fake = netG(noise)
    label = torch.full(
        (Config.batch_size,), fake_label, dtype=torch.float, device=device
    )
    output = netD(fake.detach()).view(-1)
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    return errD_fake, D_G_z1, fake


def update_generator(netG, netD, criterion, optimizerG, fake, real_label, device):
    netG.zero_grad()
    label = torch.full(
        (Config.batch_size,), real_label, dtype=torch.float, device=device
    )
    output = netD(fake).view(-1)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()
    return errG, D_G_z2


def save_fake_images(netG, fixed_noise, img_list, iters):
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    plt.figure(figsize=(8, 8))
    for i, im in enumerate(fake):
        plt.subplot(8, 8, i + 1)
        image_data = np.clip(np.transpose(im, (1, 2, 0)), 0, 1)
        plt.imshow(image_data)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    plt.savefig(f"sample/output_{iters}.png")
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
