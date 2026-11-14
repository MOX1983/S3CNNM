import torch.nn as nn

from parts_model import TwoCnnLayers, Encoder, Decoder, OutConv2d

class UNetModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down1 = Encoder(in_channels, 64)
        self.down2 = Encoder(64, 128)
        self.down3 = Encoder(128, 256)
        self.down4 = Encoder(256, 512)

        self.end = TwoCnnLayers(512, 1024)

        self.up1 = Decoder(1024, 512)
        self.up2 = Decoder(512, 256)
        self.up3 = Decoder(256, 128)
        self.up4 = Decoder(128, 64)

        self.out = OutConv2d(64, out_channels)

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x = self.end(x)
        x = self.up1(x, y4)
        x = self.up2(x, y3)
        x = self.up3(x, y2)
        x = self.up4(x, y1)
        return self.out(x)
