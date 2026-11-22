import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from .dataset import DatasetScan
from .unet import UNetModel
from .LossMet import ComboLoss, calc_dice_metric
from ..load_yml import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(config.logs), logging.StreamHandler()]
)

def train_model(epochs: int = 30, batch_size: int = 4, accum_steps: int = 4, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"use to {device}")

    df = pd.read_csv(config.to_save_dir + "\\" + config.name_new_csv)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    logging.info(f"len train: {len(train_df)}, len val: {len(val_df)}")

    train_dataset = DatasetScan(train_df, patch_size=512, probability=0.5, repeats=2, train=True)
    val_dataset = DatasetScan(val_df, patch_size=512, probability=0.5, repeats=1, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNetModel(in_channels=1, out_channels=1).to(device)
    criterion = ComboLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_dice = 0.0
    save_model_path = config.to_save_dir + "\\" + config.name_model

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        seen_train = 0
        train_tqdm = tqdm(train_loader, desc=f"Train [{epoch}/{epochs}]", leave=True)
        for step, (images, masks) in enumerate(train_tqdm):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            batch_len = images.size(0)
            train_loss += (loss.item() * accum_steps) * batch_len
            seen_train += batch_len
            train_tqdm.set_postfix(loss=f"{train_loss / seen_train:.4f}")

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        seen_val = 0

        with torch.no_grad():
            valid_tqdm = tqdm(val_loader, desc=f"Val   [{epoch}/{epochs}]", leave=True)
            for images, masks in valid_tqdm:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                dice = calc_dice_metric(outputs, masks)

                batch_len = images.size(0)
                val_loss += loss.item() * images.size(0)
                val_dice += dice * images.size(0)
                seen_val += batch_len
                valid_tqdm.set_postfix(
                    loss=f"{val_loss / seen_val:.4f}",
                    dice=f"{val_dice / seen_val:.4f}"
                )

        val_loss /= len(val_dataset)
        val_dice /= len(val_dataset)

        scheduler.step(val_dice)
        print()
        logging.info(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_model_path)
            logging.info(f"save best model {best_val_dice:.4f}")

if __name__ == "__main__":
    train_model(epochs=30, batch_size=2, accum_steps=4, lr=1e-4)