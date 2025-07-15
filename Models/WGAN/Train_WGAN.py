import os
import glob
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import autograd
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.data import DataLoader, CacheDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd

from generative.losses import PatchAdversarialLoss
from Modules_Official import Discriminator, Generator


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize a NumPy array to range [0, 1]."""
    min_val, max_val = arr.min(), arr.max()
    return np.zeros_like(arr) if max_val == min_val else (arr - min_val) / (max_val - min_val)


def build_transforms() -> Compose:
    """Create data transformation pipeline."""
    return Compose([
        LoadImaged(keys=["image", "target"]),
        EnsureChannelFirstd(keys=["image", "target"], channel_dim="no_channel"),
        ScaleIntensityd(keys=["image", "target"], minv=-1.0, maxv=1.0)
    ])


def prepare_data(base_path: Path, image_x: str, image_y: str, batch_size: int) -> DataLoader:
    """Prepare dataloader for training."""
    x_paths, y_paths = [], []

    x_paths += glob.glob(str(base_path / image_x / "*"))

    y_paths = [p.replace(image_x, image_y) for p in x_paths]
    data = [{"image": x, "target": y} for x, y in zip(x_paths, y_paths)]

    dataset = CacheDataset(data=data, transform=build_transforms())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def make_output_dirs(root: Path):
    """Create necessary output directories."""
    paths = {
        "root": root,
        "ckpt": root / "ckpt",
        "images": root / "images",
        "logs": root / "logs"
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def compute_gradient_penalty(D, real_samples, fake_samples, device) -> torch.Tensor:
    """Compute gradient penalty for WGAN-GP style regularization."""
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    interpolated_logits = D(interpolated, interpolated)
    fake = Variable(torch.ones(interpolated_logits.shape, device=device), requires_grad=False)

    gradients = autograd.grad(
        outputs=interpolated_logits,
        inputs=interpolated,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()




def train():
    # ---- Configuration ----
    base_path = Path(r"XXXX")
    image_x = "Pre"
    image_y = "AP"
    batch_size = 8
    n_epochs = 200
    plot_interval = 5
    ckpt_interval = 10
    adv_weight = 0.05
    lambda_gp = 0.1
    lr_g, lr_d = 5e-5, 1e-5

    model_name = f"Pixel2PixelGAN_{image_y}"
    output_dir = Path.cwd() / f"{model_name}_results"
    paths = make_output_dirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=paths["logs"])
    train_loader = prepare_data(base_path, image_x, image_y, batch_size)

    # ---- Model / Loss / Optimizers ----
    model = Generator(1, 1).to(device)
    discriminator = Discriminator(8, 5).to(device)

    optimizer_g = Adam(model.parameters(), lr=lr_g)
    optimizer_d = Adam(discriminator.parameters(), lr=lr_d)

    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=n_epochs, eta_min=1e-7)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=n_epochs, eta_min=1e-7)

    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    recon_loss_fn = MSELoss()

    # ---- Training Loop ----
    for epoch in range(1, n_epochs + 1):
        model.train()
        discriminator.train()
        total_recon, total_gen, total_disc = 0.0, 0.0, 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        pbar.set_description(f"[Epoch {epoch}]")

        for step, batch in pbar:
            real_x = batch["image"].to(device)
            real_y = batch["target"].to(device)

            # ---- Train Discriminator ----
            optimizer_d.zero_grad()
            fake_y = model(real_x)
            logits_real = discriminator(real_y, real_y)
            logits_fake = discriminator(fake_y.detach(), fake_y.detach())
            loss_d = 0.5 * (adv_loss_fn(logits_fake, False, True) + adv_loss_fn(logits_real, True, True))
            gp = compute_gradient_penalty(discriminator, real_y, fake_y, device)
            total_d = adv_weight * loss_d + lambda_gp * gp
            total_d.backward()
            optimizer_d.step()

            # ---- Train Generator ----
            optimizer_g.zero_grad(set_to_none=True)
            fake_y = model(real_x)
            logits_fake = discriminator(fake_y, fake_y)
            recon_loss = recon_loss_fn(fake_y, real_y)
            loss_g = recon_loss + adv_weight * adv_loss_fn(logits_fake, True, False)
            loss_g.backward()
            optimizer_g.step()

            total_recon += recon_loss.item()
            total_gen += loss_g.item()
            total_disc += loss_d.item()

            pbar.set_postfix({
                "Recon": total_recon / (step + 1),
                "Gen": total_gen / (step + 1),
                "Disc": total_disc / (step + 1),
            })

        # ---- Logging ----
        writer.add_scalar("Loss/Generator", total_gen / len(train_loader), epoch)
        writer.add_scalar("Loss/Discriminator", total_disc / len(train_loader), epoch)
        writer.add_scalar("Loss/Reconstruction", total_recon / len(train_loader), epoch)

        scheduler_g.step()
        scheduler_d.step()

        # ---- Save checkpoint ----
        if epoch % ckpt_interval == 0:
            torch.save(model.state_dict(), paths["ckpt"] / f"{epoch:03d}_G.pth")
            torch.save(discriminator.state_dict(), paths["ckpt"] / f"{epoch:03d}_D.pth")



    print("Training completed.")


if __name__ == "__main__":
    train()
