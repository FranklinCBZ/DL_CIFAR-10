
import torch
import torch.nn as nn

from model import BasicBlock, CIFARResNet


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.fc(self.pool(x))
        return x * attention


class SEBasicBlock(BasicBlock):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(in_channels, out_channels, stride)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)
        out = self.relu(out + identity)

        return out


class SEResNet(CIFARResNet):
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [
            SEBasicBlock(self.in_channels, out_channels, stride)
        ]

        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(
                SEBasicBlock(self.in_channels, out_channels)
            )

        return nn.Sequential(*layers)



class DeepSEResNet(CIFARResNet):
    def __init__(
        self,
        num_classes=10,
        base_width=32,
        blocks=(3, 3, 3, 2)
    ):
        nn.Module.__init__(self)

        self.in_channels = base_width

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                base_width,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(
            base_width, blocks[0], stride=1
        )

        self.layer2 = self._make_layer(
            base_width * 2, blocks[1], stride=2
        )

        self.layer3 = self._make_layer(
            base_width * 4, blocks[2], stride=2
        )

        self.layer4 = self._make_layer(
            base_width * 8, blocks[3], stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8, num_classes)



class DeepSEResNetDropout(DeepSEResNet):
    def __init__(
        self,
        num_classes=10,
        base_width=39,
        blocks=(3, 3, 3, 2),
        dropout=0.15
    ):
        super().__init__(
            num_classes=num_classes,
            base_width=base_width,
            blocks=blocks
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Dropout only active during training
        x = self.dropout(x)

        x = self.fc(x)
        return x



    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [
            SEBasicBlock(
                self.in_channels,
                out_channels,
                stride
            )
        ]

        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(
                SEBasicBlock(
                    self.in_channels,
                    out_channels
                )
            )

        return nn.Sequential(*layers)



if __name__ == "__main__":
    model = SEResNet(base_width=40)

    params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {params:,}")
    print(f"Under 5M limit: {params <= 5_000_000}")

    output = model(torch.randn(2, 3, 32, 32))
    print("Output shape:", output.shape)
