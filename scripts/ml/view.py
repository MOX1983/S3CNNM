import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .unet import UNetModel
from ..load_yml import config

def get_gaussian_weight(patch_size=512, sigma_scale=0.5):
    coords = np.linspace(-1, 1, patch_size)
    x, y = np.meshgrid(coords, coords)
    w = np.exp(-(x**2 + y**2) / (2 * sigma_scale**2))
    return (w + 1e-3).astype(np.float32)

def predict_full_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    patch_size: int = 512,
    stride: int = 256,
    threshold: float = 0.7
):
    orig_img = np.array(Image.open(image_path), dtype=np.float32) / 65535.0
    h, w = orig_img.shape

    pad_h = (patch_size - h % stride) % stride + patch_size
    pad_w = (patch_size - w % stride) % stride + patch_size
    padded_img = np.pad(orig_img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    ph, pw = padded_img.shape

    prob_map = np.zeros((ph, pw), dtype=np.float32)
    weight_map = np.zeros((ph, pw), dtype=np.float32)
    window_weight = get_gaussian_weight(patch_size)

    model.eval()
    with torch.no_grad():
        for y in range(0, ph - patch_size + 1, stride):
            for x in range(0, pw - patch_size + 1, stride):
                patch = padded_img[y:y + patch_size, x:x + patch_size]
                if patch.max() < 0.02:
                    continue

                tensor_patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                logits = model(tensor_patch)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()

                prob_map[y:y + patch_size, x:x + patch_size] += probs * window_weight
                weight_map[y:y + patch_size, x:x + patch_size] += window_weight

    avg_prob_map = (prob_map / np.maximum(weight_map, 1e-6))[:h, :w]
    my, mx = int(h * 0.03), int(w * 0.03)
    avg_prob_map[:my] = 0
    avg_prob_map[-my:] = 0
    avg_prob_map[:, :mx] = 0
    avg_prob_map[:, -mx:] = 0
    pred_mask = (avg_prob_map > threshold).astype(np.uint8)
    return orig_img, pred_mask, avg_prob_map

def view( test_mack_path: str, orig_img, pred_mask, prob_map):
    fig, ax = plt.subplots(1, 3, figsize=(16, 7))
    ax[0].imshow(orig_img, cmap="gray")
    test_mask = np.array(Image.open(test_mack_path))
    ax[0].imshow(test_mask, cmap="gray", alpha=0.5)

    im = ax[1].imshow(prob_map, cmap="jet", vmin=0, vmax=max(0.5, prob_map.max()))
    plt.colorbar(im, ax=ax[1], fraction=0.046)

    ax[2].imshow(orig_img, cmap="gray")
    red_overlay = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    red_overlay[pred_mask > 0] = [1.0, 0.0, 0.0, 1.0]
    ax[2].imshow(red_overlay)

    plt.tight_layout()
    plt.show()


def main(test_image_path:str, test_mack_path:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetModel(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(config.to_save_dir + "\\" + config.name_model, map_location=device, weights_only=True))
    orig_img, pred_mask, prob_map = predict_full_image(model, test_image_path, device, threshold=0.6)
    view(test_mack_path, orig_img, pred_mask, prob_map)


if __name__ == "__main__":
    test_image_path = r"D:\S3CNNM\.data\raw\ready_made_img\full\P_00162_LEFT_CC.png"
    test_mack_path = r"D:\S3CNNM\.data\raw\ready_made_img\mask\P_00162_LEFT_CC.png"
    main(test_image_path, test_mack_path)
