import torch
import torch.nn as nn


class ComboLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        num = targets.size(0)
        probs = torch.sigmoid(logits)
        x = probs.view(num, -1)
        y = targets.view(num, -1)

        intersection = (x * y).sum(1)
        dice_score = (2.0 * intersection + self.smooth) / (x.sum(1) + y.sum(1) + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return 0.5 * bce_loss + 0.5 * dice_loss


def calc_dice_metric(logits, targets, threshold=0.5, smooth=1e-5):
    probs = (torch.sigmoid(logits) > threshold).float()
    num = targets.size(0)
    x = probs.view(num, -1)
    y = targets.view(num, -1)
    intersection = (x * y).sum(1)
    dice = (2.0 * intersection + smooth) / (x.sum(1) + y.sum(1) + smooth)
    return dice.mean().item()