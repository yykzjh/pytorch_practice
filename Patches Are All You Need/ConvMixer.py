import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvBlockWithActBn(conv, h):
    return nn.Sequential(
        conv,
        nn.GELU(),
        nn.BatchNorm2d(h)
    )


class ConvMixer(nn.Module):
    def __init__(self, in_channels=3, h=1536, depth=20, kernel_size=9, patch_size=7, n_classes=1000):
        super().__init__()
        # Patch Embedding layer
        self.patch_embedding_layer = ConvBlockWithActBn(
            nn.Conv2d(in_channels=in_channels, out_channels=h, kernel_size=patch_size, stride=patch_size),
            h
        )

        # ConvMixer Layers
        self.conv_mixer_layers = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(ConvBlockWithActBn(nn.Conv2d(h, h, kernel_size, groups=h, padding="same"), h)),
                ConvBlockWithActBn(nn.Conv2d(h, h, 1), h)
            ) for i in range(depth)
        ])

        # linear layer
        self.linear_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(h, n_classes)
        )


    def forward(self, x):
        out = self.patch_embedding_layer(x)
        for layer in self.conv_mixer_layers:
            out = layer(out)
        out = self.linear_layer(out)

        return out




if __name__ == '__main__':

    model = ConvMixer()

    x = torch.randn((32, 3, 224, 224))

    output = model(x)

    print(output.size())







