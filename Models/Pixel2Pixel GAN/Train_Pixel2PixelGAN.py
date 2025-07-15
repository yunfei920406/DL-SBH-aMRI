import os
import time
import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Compose,
)

from Modules import Generator, Discriminator
from generative.losses import PatchAdversarialLoss


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Linearly normalize a NumPy array to [0,1].
    If all values are equal, returns an array of zeros.
    """
    min_val, max_val = arr.min(), arr.max()
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def build_transforms() -> Compose:
    """
    Create a MONAI compose transform for loading and scaling images.
    """
    return Compose([
        LoadImaged(keys=["image", "target"]),
        EnsureChannelFirstd(keys=["image", "target"], channel_dim="no_channel"),
        ScaleIntensityd(keys=["image", "target"], minv=-1.0, maxv=1.0),
    ])


def prepare_data(base_dir: Path, source_modality: str, target_modality: str, batch_size: int):
    """
    Scan for paired image files and return a MONAI DataLoader.
    """
    x_paths = sorted(glob.glob(str(base_dir / source_modality / "*")))
    y_paths = [p.replace(source_modality, target_modality) for p in x_paths]
    data_list = [{"image": x, "target": y} for x, y in zip(x_paths, y_paths)]
    ds = CacheDataset(data=data_list, transform=build_transforms())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def make_output_dirs(root: Path, model_name: str):
    """
    Create output directories for checkpoints, images, and logs.
    """
    paths = {
        "root": root,
        "ckpt": root / "ckpt",
        "images": root / "images",
        "logs": root / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def train_one_epoch(
    epoch: int,
    loader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    adv_loss_fn: PatchAdversarialLoss,
    recon_loss_fn: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Run a single training epoch and return (recon_loss, gen_loss, disc_loss).
    """
    generator.train()
    discriminator.train()

    total_recon, total_gen, total_disc = 0.0, 0.0, 0.0

    pbar = tqdm(enumerate(loader), total=len(loader), ncols=150)
    pbar.set_description(f"[Epoch {epoch}] Training")

    for step, batch in pbar:
        imgs = batch["image"].to(device)
        targs = batch["target"].to(device)

        # ---- Discriminator step ----
        fake = generator(imgs)
        fake_logit = discriminator(fake, fake)
        real_logit = discriminator(targs, targs)

        loss_d = 0.5 * (
            adv_loss_fn(fake_logit, target_is_real=False, for_discriminator=True)
            + adv_loss_fn(real_logit, target_is_real=True, for_discriminator=True)
        )
        optim_d.zero_grad()
        loss_d.backward()
        optim_d.step()

        # ---- Generator step ----
        fake = generator(imgs)
        fake_logit = discriminator(fake, fake)
        adv_loss = adv_loss_fn(fake_logit, target_is_real=True, for_discriminator=False)
        recon_loss = recon_loss_fn(fake, targs)
        loss_g = adv_loss + 10.0 * recon_loss  # weight factor for reconstruction

        optim_g.zero_grad()
        loss_g.backward()
        optim_g.step()

        # Accumulate metrics
        total_recon += recon_loss.item()
        total_gen += adv_loss.item()
        total_disc += loss_d.item()

        # Update progress bar
        pbar.set_postfix({
            "Recon": total_recon / (step + 1),
            "Gen": total_gen / (step + 1),
            "Disc": total_disc / (step + 1),
        })

    # Return average losses
    n = len(loader)
    return total_recon / n, total_gen / n, total_disc / n


def main():
    # ---------- Configuration ----------
    base_path = Path(XXXXX) #####Input your image path
    src_modality = "Pre"
    tgt_modality = "AP"
    batch_size = 16
    n_epochs = 200
    ckpt_interval = 10
    lr_g, lr_d = 5e-5, 1e-5

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = f"Pixel2PixelGAN_official_resnet_{tgt_modality}"
    out_dir = Path.cwd() / f"{model_name}_results"
    paths = make_output_dirs(out_dir, model_name)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=paths["logs"])

    # Data loader
    train_loader = prepare_data(base_path, src_modality, tgt_modality, batch_size)

    # Models
    generator = Generator(in_channels=1, out_channels=1).to(device)
    discriminator = Discriminator(in_channels=8, n_blocks=5).to(device)

    # Losses and optimizers
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    recon_loss_fn = nn.MSELoss()
    optim_g = Adam(generator.parameters(), lr=lr_g)
    optim_d = Adam(discriminator.parameters(), lr=lr_d)

    # LR schedulers
    sched_g = CosineAnnealingLR(optim_g, T_max=n_epochs, eta_min=1e-7)
    sched_d = CosineAnnealingLR(optim_d, T_max=n_epochs, eta_min=1e-7)

    # Training loop
    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        recon_loss, gen_loss, disc_loss = train_one_epoch(
            epoch, train_loader,
            generator, discriminator,
            optim_g, optim_d,
            adv_loss_fn, recon_loss_fn,
            device
        )

        # Log to TensorBoard
        writer.add_scalar("Loss/Generator", gen_loss, epoch)
        writer.add_scalar("Loss/Discriminator", disc_loss, epoch)
        writer.add_scalar("Loss/Reconstruction", recon_loss, epoch)

        # Step schedulers
        sched_g.step()
        sched_d.step()

        # Save checkpoints periodically
        if epoch % ckpt_interval == 0:
            torch.save(generator.state_dict(), paths["ckpt"] / f"{epoch:03d}_G.pth")
            torch.save(discriminator.state_dict(), paths["ckpt"] / f"{epoch:03d}_D.pth")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60:.2f} minutes.")


if __name__ == "__main__":
    main()
