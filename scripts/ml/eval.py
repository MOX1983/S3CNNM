import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from .unet import UNetModel
from ..load_yml import config


def get_gaussian_weight(patch_size=512, sigma_scale=0.5):
    coords = np.linspace(-1, 1, patch_size)
    x, y = np.meshgrid(coords, coords)
    w = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_scale ** 2))
    return (w + 1e-3).astype(np.float32)


@torch.no_grad()
def predict_prob_map(model, img, device, patch_size=512, stride=256):
    h, w = img.shape
    pad_h = (patch_size - h % stride) % stride + patch_size
    pad_w = (patch_size - w % stride) % stride + patch_size
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    ph, pw = padded.shape

    prob = np.zeros((ph, pw), np.float32)
    wsum = np.zeros((ph, pw), np.float32)
    win = get_gaussian_weight(patch_size)

    model.eval()
    for y in range(0, ph - patch_size + 1, stride):
        for x in range(0, pw - patch_size + 1, stride):
            patch = padded[y:y + patch_size, x:x + patch_size]
            if patch.max() < 0.02:
                continue
            t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            p = torch.sigmoid(model(t)).squeeze().cpu().numpy()
            prob[y:y + patch_size, x:x + patch_size] += p * win
            wsum[y:y + patch_size, x:x + patch_size] += win

    prob = (prob / np.maximum(wsum, 1e-6))[:h, :w]
    my, mx = max(1, int(h * 0.03)), max(1, int(w * 0.03))
    prob[:my] = 0
    prob[-my:] = 0
    prob[:, :mx] = 0
    prob[:, -mx:] = 0
    return prob


def region_metrics(pred, gt, min_area):
    lab, n = ndi.label(pred)
    if n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_area
        keep[0] = False
        pred = keep[lab]
        lab, n = ndi.label(pred)
    gt_lab, n_gt = ndi.label(gt)
    hit_pred = set(np.unique(lab[gt > 0]).tolist()) - {0}
    hit_gt = set(np.unique(gt_lab[pred > 0]).tolist()) - {0}
    return pred, {"n_pred": n, "fp": n - len(hit_pred), "n_gt": n_gt, "tp_gt": len(hit_gt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    ap.add_argument("--min_area", type=int, default=100, help="мин. площадь предсказанной области, пикс.")
    ap.add_argument("--limit", type=int, default=None, help="взять только первые N val-снимков")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(config.to_save_dir, config.name_model)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = UNetModel(1, 1).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    df = pd.read_csv(os.path.join(config.to_save_dir, config.name_new_csv))
    _, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                      .split(df, groups=df["patient_id"]))
    val_df = df.iloc[val_idx].drop_duplicates(subset="full").reset_index(drop=True)
    if a.limit:
        val_df = val_df.head(a.limit)
    print(f"val-снимков: {len(val_df)}")

    records, resized = [], 0
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        img = np.array(Image.open(row["full"]), dtype=np.float32) / 65535.0
        m_img = Image.open(row["mask"])
        if m_img.size != (img.shape[1], img.shape[0]):
            m_img = m_img.resize((img.shape[1], img.shape[0]), Image.NEAREST)
            resized += 1
        gt = np.array(m_img) > 0

        prob = predict_prob_map(model, img, device)
        for thr in a.thr:
            pred, reg = region_metrics(prob > thr, gt, a.min_area)
            records.append({
                "name": os.path.basename(row["full"]), "thr": thr,
                "inter": int((pred & gt).sum()), "pred_px": int(pred.sum()), "gt_px": int(gt.sum()),
                **reg,
            })

    per_image = pd.DataFrame(records)
    per_image["dice"] = 2 * per_image["inter"] / (per_image["pred_px"] + per_image["gt_px"] + 1e-6)
    per_image["found"] = per_image["inter"] > 0

    rows = []
    for thr, g in per_image.groupby("thr"):
        rows.append({
            "порог": thr,
            "Dice (все пиксели)": 2 * g["inter"].sum() / (g["pred_px"].sum() + g["gt_px"].sum() + 1e-6),
            "Dice (среднее по снимкам)": g["dice"].mean(),
            "снимков с найденным очагом, %": 100 * g["found"].mean(),
            "найдено очагов, %": 100 * g["tp_gt"].sum() / max(g["n_gt"].sum(), 1),
            "ложных областей на снимок": g["fp"].mean(),
            "точность по областям, %": 100 * (g["n_pred"].sum() - g["fp"].sum()) / max(g["n_pred"].sum(), 1),
        })
    summary = pd.DataFrame(rows).round(3)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n" + summary.to_string(index=False))

    out_dir = config.to_save_dir
    per_image.to_csv(os.path.join(out_dir, "eval_per_image.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "eval_summary.csv"), index=False)

if __name__ == "__main__":
    main()