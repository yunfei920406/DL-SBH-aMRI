import os
import glob
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from monai.data import DataLoader, CacheDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd
from generative.losses import PatchAdversarialLoss

from cGAN.Modules_cGAN_official import Discriminator, ConditionalGenerator as Generator


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize NumPy array to [0, 1] range."""
    min_val, max_val = arr.min(), arr.max()
    return np.zeros_like(arr) if max_val == min_val else (arr - min_val) / (max_val - min_val)


def build_transforms() -> Compose:
    """Create image transformations pipeline."""
    return Compose([
        LoadImaged(keys=["image", "target"]),
        EnsureChannelFirstd(keys=["image", "target"], channel_dim="no_channel"),
        ScaleIntensityd(keys=["image", "target"], minv=-1, maxv=1),
    ])


def prepare_data(base_path: Path, image_x: str, image_y_list: list, category_map: dict, batch_size: int):
    """
    Load and transform dataset for conditional GAN training.
    """
    x_paths = []
    # Collect data from Sheshan and Internal datasets

    x_paths = (glob.glob(str(base_path / image_x / "*")))

    # Duplicate x_paths for each modality
    x_paths *= len(image_y_list)

    # Construct corresponding y_paths
    y_paths = []
    for image_y in image_y_list:
        y_paths.extend([x.replace(image_x, image_y) for x in x_paths[:len(x_paths)//len(image_y_list)]])

    # Pack into dictionary with category label
    data = []
    for x, y in zip(x_paths, y_paths):
        label = Path(y).parent.name  # Use folder name as label
        category = category_map.get(label, -1)
        data.append({"image": x, "target": y, "category": category})

    dataset = CacheDataset(data=data, transform=build_transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def make_output_dirs(root: Path):
    """
    Create directories for checkpoints, images, and logs.
    """
    paths = {
        "root": root,
        "ckpt": root / "ckpt",
        "images": root / "images",
        "logs": root / "logs"
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def train_one_epoch(epoch, loader, model, discriminator, optim_g, optim_d, adv_loss_fn, recon_loss_fn, device):
    """
    Perform one training epoch of the conditional GAN.
    """
    model.train()
    discriminator.train()
    total_recon, total_gen, total_disc = 0.0, 0.0, 0.0

    pbar = tqdm(enumerate(loader), total=len(loader), ncols=150)
    pbar.set_description(f"[Epoch {epoch}]")

    for step, batch in pbar:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        categories = batch["category"].to(device)

        # ---- Train Discriminator ----
        fake = model(images, categories)
        fake_logits = discriminator(fake, categories)
        real_logits = discriminator(targets, categories)
        loss_d = 0.5 * (
            adv_loss_fn(fake_logits, target_is_real=False, for_discriminator=True) +
            adv_loss_fn(real_logits, target_is_real=True, for_discriminator=True)
        )
        optim_d.zero_grad()
        loss_d.backward()
        optim_d.step()

        # ---- Train Generator ----
        fake = model(images, categories)
        fake_logits = discriminator(fake, categories)
        loss_adv = adv_loss_fn(fake_logits, target_is_real=True, for_discriminator=False)
        loss_recon = recon_loss_fn(fake, targets)
        loss_g = loss_adv + 10 * loss_recon  # Weighted loss

        optim_g.zero_grad()
        loss_g.backward()
        optim_g.step()

        total_recon += loss_recon.item()
        total_gen += loss_adv.item()
        total_disc += loss_d.item()

        pbar.set_postfix({
            "Recon": total_recon / (step + 1),
            "Gen": total_gen / (step + 1),
            "Disc": total_disc / (step + 1),
        })

    n = len(loader)
    return total_recon / n, total_gen / n, total_disc / n


def main():
    # ---- Configuration ----
    base_path = Path(r"XXXXXXX")
    image_x = "Pre"
    image_y_list = ["AP", "VP", "DP", "T2", "DWI_b600", "ADC"]
    category_map = {name: i for i, name in enumerate(image_y_list)}

    target_modality = "AP"  # Used only for naming and logging
    model_name = f"cGAN_{target_modality}"
    out_root = Path.cwd() / f"{model_name}_results"
    batch_size = 8
    n_epochs = 200
    ckpt_interval = 10
    lr_g, lr_d = 5e-5, 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = make_output_dirs(out_root)
    writer = SummaryWriter(log_dir=paths["logs"])

    # ---- Data ----
    train_loader = prepare_data(base_path, image_x, image_y_list, category_map, batch_size)

    # ---- Models ----
    model = Generator(in_channels=1, out_channels=1).to(device)
    discriminator = Discriminator(in_channels=8, n_blocks=5).to(device)

    # ---- Loss, optimizer, scheduler ----
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    recon_loss_fn = nn.MSELoss()
    optim_g = Adam(model.parameters(), lr=lr_g)
    optim_d = Adam(discriminator.parameters(), lr=lr_d)
    sched_g = CosineAnnealingLR(optim_g, T_max=n_epochs, eta_min=1e-7)
    sched_d = CosineAnnealingLR(optim_d, T_max=n_epochs, eta_min=1e-7)

    # ---- Training loop ----
    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        recon_loss, gen_loss, disc_loss = train_one_epoch(
            epoch, train_loader, model, discriminator,
            optim_g, optim_d, adv_loss_fn, recon_loss_fn, device
        )

        # Logging
        writer.add_scalar("Loss/Generator", gen_loss, epoch)
        writer.add_scalar("Loss/Discriminator", disc_loss, epoch)
        writer.add_scalar("Loss/Reconstruction", recon_loss, epoch)

        sched_g.step()
        sched_d.step()

        # Save checkpoint
        if epoch % ckpt_interval == 0:
            torch.save(model.state_dict(), paths["ckpt"] / f"{epoch:03d}_G.pth")
            torch.save(discriminator.state_dict(), paths["ckpt"] / f"{epoch:03d}_D.pth")


if __name__ == "__main__":
    main()
