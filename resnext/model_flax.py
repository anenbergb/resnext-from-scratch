import jax
from flax import nnx
from functools import partial

"""
JAX/Flax assumes tensors are in NHWC order
Pytorch assumes tensors are in NCHW order
"""


class Bottleneck(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        mult_factor: int = 2,
        stride: int = 1,
        stage_index: int = 1,
        cardinality: int = 32,
        d: int = 4,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """
        hidden_dim = stage_index * d * cardinality
            e.g. stage_index = 1, d = 4, cardinality = 32, hidden_dim = 128
        out_channels = hidden_dim * mult_factor
            e.g. hidden_dim = 128, mult_factor = 2, out_channels = 256
        """
        assert mult_factor > 0, "Multiplication factor must be greater than 0"
        assert stage_index > 0, "stage index must be greater than 0"
        assert cardinality > 0, "Cardinality must be greater than 0"
        assert d > 0, "Width of each group must be greater than 0"

        hidden_dim = stage_index * d * cardinality
        out_channels = hidden_dim * mult_factor

        self.main_branch = nnx.Dict(
            {
                "conv1": nnx.Conv(
                    in_channels,
                    hidden_dim,
                    kernel_size=(1, 1),
                    use_bias=False,
                    rngs=rngs,
                ),
                # assumes input tensor is (N,H,W,C). BN should normalize over all axes except the last.
                "bn1": nnx.BatchNorm(hidden_dim, rngs=rngs),
                "relu1": nnx.relu,  # not in place
                "conv2": nnx.Conv(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=(3, 3),
                    strides=(stride, stride),
                    use_bias=False,
                    feature_group_count=cardinality,
                    # padding = "SAME" is equivalent to padding = 1
                    rngs=rngs,
                ),
                "bn2": nnx.BatchNorm(hidden_dim, rngs=rngs),
                "relu2": nnx.relu,
                "conv3": nnx.Conv(
                    hidden_dim,
                    out_channels,
                    kernel_size=(1, 1),
                    use_bias=False,
                    rngs=rngs,
                ),
                "bn3": nnx.BatchNorm(out_channels, rngs=rngs),
            }
        )
        self.downsample = nnx.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nnx.Sequential(
                nnx.Conv(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    strides=(stride, stride),
                    use_bias=False,
                    rngs=rngs,
                ),
                nnx.BatchNorm(out_channels, rngs=rngs),
            )

        self.in_channels = in_channels
        self.mult_factor = mult_factor
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.cardinality = cardinality
        self.d = d
        self.stage_index = stage_index

    def __call__(self, x):
        out = x
        for layer in self.main_branch.values():
            out = layer(out)
        out += self.downsample(x)
        out = nnx.relu(out)
        return out


class ResNeXt(nnx.Module):
    def __init__(
        self,
        num_layers: list[int] = [3, 4, 6, 3],
        in_channels: int = 3,
        stem_channels: int = 64,
        num_classes: int = 1000,
        mult_factor: int = 2,
        cardinality: int = 32,
        d: int = 4,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.num_classes = num_classes
        self.mult_factor = mult_factor
        self.cardinality = cardinality
        self.d = d

        self.stem = nnx.Sequential(
            nnx.Conv(
                in_channels,
                stem_channels,
                kernel_size=(7, 7),
                strides=(2, 2),
                use_bias=False,
                # padding = "SAME" is equivalent to padding = 3
                rngs=rngs,
            ),
            nnx.BatchNorm(stem_channels, rngs=rngs),
            nnx.relu,
            partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2), padding="SAME"),
            # padding = "SAME" is equivalent to padding = 1
        )

        in_channels = stem_channels
        self.stages = nnx.Dict()
        for stage_index, layers_per_stage in enumerate(num_layers):
            stage, in_channels = self.make_stage(
                in_channels, stage_index + 1, layers_per_stage, rngs=rngs
            )
            self.stages.update({f"stage{stage_index}": stage})

        self.fc = nnx.Linear(in_channels, num_classes, use_bias=True, rngs=rngs)

    def make_stage(
        self,
        in_channels: int,
        stage_index: int,
        layers_per_stage: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
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
                    rngs=rngs,
                )
            )
            in_channels = layers[-1].out_channels
        return nnx.Sequential(*layers), in_channels

    def __call__(self, x):
        out = self.stem(x)
        for stage in self.stages.values():
            out = stage(out)
        # (N,H,W,C)
        # adaptive average pool
        out = jax.numpy.mean(out, axis=(1, 2), keepdims=False)  # (N,1,C)
        out = self.fc(out)
        return out
