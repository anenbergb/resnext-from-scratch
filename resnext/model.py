import torch
from torch import nn


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNeXt.

    Args:
        in_channels (int): Number of input channels.
        mult_factor (int): Multiplication factor for the bottleneck layers.
        stride (int): Stride for the convolutional layers.
        stage_index (int): Index of the current layer.
        cardinality (int): Number of groups in the grouped convolution.
        d (int): Width of each group in the grouped convolution.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization for the first convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization for the second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        bn3 (nn.BatchNorm2d): Batch normalization for the third convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        downsample (nn.Sequential): Downsample layer to match dimensions.
        main_branch (nn.Sequential): Main branch of the bottleneck block.
        in_channels (int): Number of input channels.
        mult_factor (int): Multiplication factor for the bottleneck layers.
        hidden_dim (int): Hidden dimension of the bottleneck block.
        out_channels (int): Number of output channels.
        cardinality (int): Number of groups in the grouped convolution.
        d (int): Width of each group in the grouped convolution.
        stage_index (int): Index of the current layer.
    """

    def __init__(
        self, in_channels, mult_factor=2, stride=1, stage_index=1, cardinality=32, d=4
    ):
        """
        hidden_dim = stage_index * d * cardinality
            e.g. stage_index = 1, d = 4, cardinality = 32, hidden_dim = 128
        out_channels = hidden_dim * mult_factor
            e.g. hidden_dim = 128, mult_factor = 2, out_channels = 256
        """
        super(Bottleneck, self).__init__()
        assert mult_factor > 0, "Multiplication factor must be greater than 0"
        assert stage_index > 0, "stage index must be greater than 0"
        assert cardinality > 0, "Cardinality must be greater than 0"
        assert d > 0, "Width of each group must be greater than 0"

        hidden_dim = stage_index * d * cardinality
        out_channels = hidden_dim * mult_factor

        self.main_branch = nn.ModuleDict(
            {
                "conv1": nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                "bn1": nn.BatchNorm2d(hidden_dim),
                "relu1": nn.ReLU(inplace=True),
                "conv2": nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                    groups=cardinality,
                ),
                "bn2": nn.BatchNorm2d(hidden_dim),
                "relu2": nn.ReLU(inplace=True),
                "conv3": nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                "bn3": nn.BatchNorm2d(out_channels),
            }
        )
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

        self.in_channels = in_channels
        self.mult_factor = mult_factor
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.cardinality = cardinality
        self.d = d
        self.stage_index = stage_index

    def forward(self, x):
        out = x
        for layer in self.main_branch.values():
            out = layer(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):
    """
    ResNeXt model implementation.
    resnext50_32x4d = ResNeXt(num_layers=[3, 4, 6, 3], in_channels=3, stem_channels=64, num_classes=1000, mult_factor=2, cardinality=32, d=4)
    resnext101_32x8d = ResNeXt(num_layers=[3, 4, 23, 3], in_channels=3, stem_channels=64, num_classes=1000, mult_factor=1, cardinality=32, d=8)

    Args:
        num_layers (list): Number of layers in each block.
        in_channels (int): Number of input channels.
        stem_channels (int): Number of channels in the stem layer.
        num_classes (int): Number of output classes.
        mult_factor (int): Multiplication factor for the bottleneck layers.
        cardinality (int): Number of groups in the grouped convolution.
        d (int): Width of each group in the grouped convolution.

    Attributes:
        num_layers (list): Number of layers in each block.
        in_channels (int): Number of input channels.
        stem_channels (int): Number of channels in the stem layer.
        num_classes (int): Number of output classes.
        mult_factor (int): Multiplication factor for the bottleneck layers.
        cardinality (int): Number of groups in the grouped convolution.
        d (int): Width of each group in the grouped convolution.
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization for the first convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        maxpool (nn.MaxPool2d): Max pooling layer.
        stem (nn.Sequential): Stem layer consisting of conv1, bn1, relu, and maxpool.
        avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        fc (nn.Linear): Fully connected layer for classification.
    """

    def __init__(
        self,
        num_layers=[3, 4, 6, 3],
        in_channels=3,
        stem_channels=64,
        num_classes=1000,
        mult_factor=2,
        cardinality=32,
        d=4,
    ):
        super(ResNeXt, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.num_classes = num_classes
        self.mult_factor = mult_factor
        self.cardinality = cardinality
        self.d = d

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        in_channels = stem_channels
        self.stages = nn.ModuleDict()
        for stage_index, layers_per_stage in enumerate(num_layers):
            stage = self.make_stage(in_channels, stage_index + 1, layers_per_stage)
            self.stages.update({f"stage{stage_index}": stage})
            in_channels = stage[-1].out_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_stage(self, in_channels, stage_index, layers_per_stage):
        layers = []
        for i in range(layers_per_stage):
            layers.append(
                Bottleneck(
                    in_channels=in_channels,
                    mult_factor=self.mult_factor,
                    stride=2 if i == 0 and stage_index > 0 else 1,
                    stage_index=stage_index,
                    cardinality=self.cardinality,
                    d=self.d,
                )
            )
            in_channels = layers[-1].out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        for stage in self.stages.values():
            out = stage(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
