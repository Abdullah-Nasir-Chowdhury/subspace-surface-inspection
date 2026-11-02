"""Trains DentNet on the synthetic dataset produced by main_real_scale.py.

The original training script used to produce best_dentnet_real_scale.pth was
not kept alongside the rest of this project, so this file rebuilds it from
the settings recorded in config.py and research_log.md: Adam at lr=0.001,
BCEWithLogitsLoss over a two-channel one-hot target (channel 0 is normal
surface, channel 1 is dent), batch size 4, 100 epochs. Preprocessing mirrors
synthetic/s1_eval_val_iou.py exactly, since a mismatch between training
and inference preprocessing was the single biggest source of bugs during the
original development of this project.

Usage:
    python train.py

Expects samples at ~/data/pointcloud_db_real_scale, named 00000_input.npy /
00000_gt.npy and so on, as produced by synthetic/gen_parallel.py. Samples
0 to 23999 are used for training and 24000 to 29999 are held out, matching
the split used everywhere else in this project.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from model import DentNet

DATA_DIR = os.path.expanduser("~/data/pointcloud_db_real_scale")
CHECKPOINT_PATH = "best_dentnet_real_scale.pth"
TRAIN_START, TRAIN_END = 0, 24000
VAL_START, VAL_END = 24000, 30000
VAL_CHECK_SIZE = 300  # samples drawn each epoch to track progress; the full
                       # held-out set is scored properly afterward by
                       # synthetic/s1_eval_val_iou.py


def pad_to_16(array, mode="edge"):
    h, w = array.shape
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    return np.pad(array, ((0, ph), (0, pw)), mode=mode)


class DentDataset(Dataset):
    def __init__(self, data_dir, start, end):
        self.data_dir = data_dir
        self.ids = [i for i in range(start, end)
                    if os.path.exists(f"{data_dir}/{i:05d}_gt.npy")]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        residual = np.load(f"{self.data_dir}/{i:05d}_input.npy")
        mask = np.load(f"{self.data_dir}/{i:05d}_gt.npy").astype(np.float32)
        residual = pad_to_16(residual)
        mask = pad_to_16(mask)
        target = np.stack([1.0 - mask, mask], axis=0)
        return torch.from_numpy(residual).float()[None], torch.from_numpy(target).float()


def evaluate(model, dataset, device, n=VAL_CHECK_SIZE):
    model.eval()
    idx = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    ious = []
    with torch.no_grad():
        for i in idx:
            x, y = dataset[i]
            x = x[None].to(device)
            probs = torch.softmax(model(x), dim=1)
            pred = (probs[0, 1] > 0.5).cpu().numpy()
            gt = y[1].numpy() > 0.5
            inter = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            ious.append(inter / union if union else 1.0)
    model.train()
    return float(np.mean(ious))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}")

    train_set = DentDataset(DATA_DIR, TRAIN_START, TRAIN_END)
    val_set = DentDataset(DATA_DIR, VAL_START, VAL_END)
    print(f"{len(train_set)} training samples, {len(val_set)} held out")

    loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                         num_workers=4, drop_last=True)

    model = DentNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_iou = -1.0
    for epoch in range(config.NUM_EPOCHS):
        running_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        mean_loss = running_loss / max(len(loader), 1)
        val_iou = evaluate(model, val_set, device)
        print(f"epoch {epoch + 1}/{config.NUM_EPOCHS}  loss {mean_loss:.4f}  "
              f"held-out IoU (sampled) {val_iou:.4f}", flush=True)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  saved checkpoint, held-out IoU {val_iou:.4f}", flush=True)

    print("training finished, run synthetic/s1_eval_val_iou.py for the "
          "full held-out evaluation")


if __name__ == "__main__":
    main()
