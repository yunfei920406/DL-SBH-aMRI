import os
import glob
import yaml
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from monai.config import print_config
from monai.apps import DecathlonDataset
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Compose,
)

from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from LiDiffNet.DiffusionModel import DiffusionModelUNet

from generative.networks.nets import DiffusionModelUNet as DiffusionModelUNet2

# Print MONAI and PyTorch configuration info
print_config()

# Load configuration file
with open("config_ddpm.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract training parameters from config
base_path = config['base_path']
image_x = config['image_x']
image_y = config['image_y']
batch_size = config['batch_size']
lr = config['lr']
n_epochs = config['n_epochs']
start_epoch = config['start_epoch']
current_model = config['current_model']
current_date = datetime.now().strftime('%Y%m%d') if config['current_date'] == "current_date" else config['current_date']
ckpt_model = config['ckpt_model']
ckpt_save_interval = config['ckpt_save_interval']
device = torch.device(config['device'])

# File and folder naming
model_name = f"{current_model}_{current_date}.pt"
image_name = f"{current_model}_{current_date}.png"
ckpt_name = f"{current_model}_{current_date}"
output_dir = os.path.join(os.getcwd(), f"results_{current_model}_{current_date}")

ckpt_dir = os.path.join(output_dir, "ckpt")
image_dir = os.path.join(output_dir, "image")
log_dir = os.path.join(output_dir, f"{current_model}_log_{current_date}")
ckpt_path = os.path.join(output_dir, ckpt_name)

# Create required directories if they do not exist
for directory in [output_dir, ckpt_dir, image_dir, log_dir, ckpt_path]:
    os.makedirs(directory, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

def normalize_array(arr):
    """Normalize a NumPy array to range [0, 1]."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(arr)

# Prepare training data paths
image_paths = glob.glob(os.path.join(base_path, "*"))
label_paths = [p.replace(image_x, image_y) for p in image_paths]
train_datalist = [{"image": img, "target": label} for img, label in zip(image_paths, label_paths)]

# Data transformations
data_transforms = Compose([
    LoadImaged(keys=["image", "target"]),
    EnsureChannelFirstd(keys=["image", "target"], channel_dim="no_channel"),
    ScaleIntensityd(keys=["image", "target"]),
])

# Load dataset
train_dataset = CacheDataset(data=train_datalist, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Initialize U-Net model
model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,
    out_channels=1,
    num_channels=(64, 128, 256),
    attention_levels=(False, False, True),
    num_res_blocks=1,
    num_head_channels=(0, 128, 256),
)







# Load pretrained checkpoint if provided
if ckpt_model is not None:
    try:
        model.load_state_dict(torch.load(ckpt_model))
    except Exception as e:
        print("Failed to load checkpoint. Please verify path. Error:", e)

model.to(device)

# Initialize diffusion scheduler and inferer
scheduler = DDPMScheduler(num_train_timesteps=1000)
inferer = DiffusionInferer(scheduler)

# Optimizer and AMP scaler
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
scaler = GradScaler()

# Training loop
epoch_loss_list = []

for epoch in range(start_epoch, n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=80)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)

        # Sample random timestep
        timesteps = torch.randint(0, 1000, (len(images),)).to(device)
        noise = torch.randn_like(targets).to(device)

        # Add noise to the segmentation masks
        noisy_targets = scheduler.add_noise(original_samples=targets, noise=noise, timesteps=timesteps)

        # Concatenate input image and noisy target for conditioning
        combined_input = torch.cat((images, noisy_targets), dim=1)

        with autocast(enabled=True):
            predictions = model(x=combined_input, timesteps=timesteps)
            loss = F.mse_loss(predictions.float(), noise.float())

        # Backpropagation with AMP
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        writer.add_scalar('Loss/Iterative_Loss', loss.item(), epoch * len(train_loader) + step)

    # Logging epoch-level loss
    avg_epoch_loss = epoch_loss / (step + 1)
    writer.add_scalar('Loss/Epoch_Loss', avg_epoch_loss, epoch)
    epoch_loss_list.append(avg_epoch_loss)

    # Save model checkpoint periodically
    if (epoch + 1) % ckpt_save_interval == 0:
        checkpoint_filename = os.path.join(ckpt_dir, f"{epoch + 1}_{model_name}")
        torch.save(model.state_dict(), checkpoint_filename)
