import torch
import torch.nn as nn
import torch.nn.functional as F

class LossUnet(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        x = probs.view(num, -1)
        y = targets.view(num, -1)
        intersection = (x * y)

        score = 2 * (intersection.sum(1) + self.smooth) / (x.sum(1) + y.sum(1) + self.smooth)
        return 1 - score.sum() / num
