
import torch
import torch.nn as nn

from model_se import DeepSEResNetDropout, SEBlock


class PreActSEBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()

        # Pre-activation: BN -> ReLU -> Conv
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Preserve V11's SE attention
        self.se = SEBlock(out_channels)

        self.needs_projection = (
            stride != 1 or in_channels != out_channels
        )

        if self.needs_projection:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # First pre-activation
        out = self.relu1(self.bn1(x))

        # Projection uses pre-activated features;
        # identity shortcut leaves x unchanged.
        if self.needs_projection:
            identity = self.shortcut(out)
        else:
            identity = x

        # Residual branch
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)

        # Apply SE channel attention
        out = self.se(out)

        # IMPORTANT: No ReLU after addition
        return out + identity


class PreActDeepSEResNetDropout(
    DeepSEResNetDropout
):
    def _make_layer(
        self,
        out_channels,
        num_blocks,
        stride
    ):
        layers = [
            PreActSEBasicBlock(
                self.in_channels,
                out_channels,
                stride
            )
        ]

        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(
                PreActSEBasicBlock(
                    self.in_channels,
                    out_channels
                )
            )

        return nn.Sequential(*layers)


if __name__ == "__main__":
    model = PreActDeepSEResNetDropout(
        num_classes=10,
        base_width=39,
        blocks=(3, 3, 3, 2),
        dropout=0.15
    )

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {params:,}")
    print(f"Remaining budget: {5_000_000 - params:,}")
    print("Under 5M limit:", params <= 5_000_000)

    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 32, 32))

    print("Output shape:", output.shape)

    assert params <= 5_000_000
    assert output.shape == (2, 10)
