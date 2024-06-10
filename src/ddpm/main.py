# main_script.py

import copy
import json
import os
import warnings

import numpy as np
import torch
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import trange
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from datasets import load_dataset
from config import Config
from fid import get_fid_score
from ..dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x["image"]


def warmup_lr(step):
    return min(step, Config.warmup) / Config.warmup


def evaluate(sampler, model, reference_images=None):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, Config.num_images, Config.batch_size, desc=desc):
            batch_size = min(Config.batch_size, Config.num_images - i)
            x_T = torch.randn((batch_size, 3, Config.img_size, Config.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            save_image(
                torch.tensor(batch_images),
                os.path.join(Config.logdir, "sample_eval", f"sample_{i}.png"),
                nrow=16,
            )

            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()

    if reference_images is None:

        dataloader = get_dataloader(
            dataset_name=Config.dataset_name,
            image_size=Config.img_size,
            batch_size=Config.batch_size,
        )
    reference_images = next(iter(dataloader))["image"][: Config.num_images].numpy()

    FID = get_fid_score(
        Config.fid_cache,
        images,
        num_images=Config.num_images,
        use_torch=Config.fid_use_torch,
        verbose=True,
        reference_images=reference_images,
    )
    return FID, images


def train():
    dataloader = get_dataloader(
        dataset_name=Config.dataset_name,
        image_size=Config.img_size,
        batch_size=Config.batch_size,
    )

    net_model = UNet(
        T=Config.T,
        ch=Config.ch,
        ch_mult=Config.ch_mult,
        attn=Config.attn,
        num_res_blocks=Config.num_res_blocks,
        dropout=Config.dropout,
    )
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=Config.lr)
    trainer = GaussianDiffusionTrainer(
        net_model, Config.beta_1, Config.beta_T, Config.T
    ).to(device)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, warmup_lr)
    net_sampler = GaussianDiffusionSampler(
        net_model,
        Config.beta_1,
        Config.beta_T,
        Config.T,
        Config.img_size,
        Config.mean_type,
        Config.var_type,
    ).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model,
        Config.beta_1,
        Config.beta_T,
        Config.T,
        Config.img_size,
        Config.mean_type,
        Config.var_type,
    ).to(device)
    if Config.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    if not os.path.exists(Config.logdir):
        os.makedirs(os.path.join(Config.logdir, "sample"))

    x_T = torch.randn(Config.sample_size, 3, Config.img_size, Config.img_size)

    for step in trange(Config.total_steps):
        x = next(infiniteloop(dataloader)).to(device)
        loss = trainer(x)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), Config.grad_clip)
        optim.step()
        sched.step()

        ema(net_model, ema_model, Config.ema_decay)

        if step % Config.sample_step == 0:
            with torch.no_grad():
                net_model.eval()
                ema_model.eval()
                sample = net_sampler(x_T.to(device)).cpu()
                save_image(
                    torch.tensor(sample),
                    os.path.join(Config.logdir, "sample", f"sample_{step}.png"),
                    nrow=8,
                )
                net_model.train()
                ema_model.train()

        if step % Config.save_step == 0:
            torch.save(
                {
                    "model": net_model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                },
                os.path.join(Config.logdir, "model.pt"),
            )


def main():
    if Config.train:
        train()
    else:
        sampler = GaussianDiffusionSampler(
            UNet(
                T=Config.T,
                ch=Config.ch,
                ch_mult=Config.ch_mult,
                attn=Config.attn,
                num_res_blocks=Config.num_res_blocks,
                dropout=Config.dropout,
            ),
            Config.beta_1,
            Config.beta_T,
            Config.T,
            Config.img_size,
            Config.mean_type,
            Config.var_type,
        ).to(device)
        state_dict = torch.load(os.path.join(Config.logdir, "model.pt"))
        sampler.load_state_dict(state_dict["model"])
        FID, images = evaluate(sampler, state_dict["ema"])
        print(f"FID: {FID}")
        save_image(
            torch.tensor(images),
            os.path.join(Config.logdir, "sample_eval", f"sample.png"),
            nrow=16,
        )
        print(f"Saved images to {os.path.join(Config.logdir, 'sample_eval')}")


if "__name__" == "__main__":
    main()
