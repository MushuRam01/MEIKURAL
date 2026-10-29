"""Larger CNN for the full native STFT image, with no frequency-axis resizing."""

from torch import nn

from ...ver1.training.model import select_device
from ..config import feature_shape

__all__ = ["RaagCNN", "model_dimensions", "select_device"]


class RaagCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        m = config["model"]
        blocks = []
        incoming = 1
        for channels, kernel in zip(m["channels"], m["kernels"], strict=True):
            blocks.extend([nn.Conv2d(incoming, channels, tuple(kernel), padding="same"),
                           nn.ReLU(), nn.MaxPool2d(tuple(m["pool"]))])
            incoming = channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(tuple(m["adaptive_pool"]))
        flattened = incoming * m["adaptive_pool"][0] * m["adaptive_pool"][1]
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(m["dropout"]),
            nn.Linear(flattened, m["hidden_features"]), nn.ReLU(), nn.Dropout(m["dropout"]),
            nn.Linear(m["hidden_features"], len(config["classes"])))

    def forward(self, matrix):
        return self.head(self.pool(self.features(matrix)))


def model_dimensions(config):
    """Derive shapes/RFs from actual layers without allocating large activations."""
    model = RaagCNN(config)
    height, width = feature_shape(config)
    channels = 1
    receptive, jump = [1, 1], [1, 1]
    rows = [{"layer": "Input", "shape": [channels, height, width],
             "parameters": 0, "receptive_field": receptive.copy()}]
    for index, layer in enumerate(model.features):
        if not isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            continue
        kernel = layer.kernel_size
        stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
        if isinstance(layer, nn.Conv2d):
            channels = layer.out_channels
        else:
            height //= stride[0]
            width //= stride[1]
        for axis in range(2):
            receptive[axis] += (kernel[axis] - 1) * jump[axis]
            jump[axis] *= stride[axis]
        rows.append({"layer": f"{type(layer).__name__} {index // 3 + 1}",
                     "shape": [channels, height, width],
                     "parameters": sum(p.numel() for p in layer.parameters()),
                     "receptive_field": receptive.copy()})
    rows.extend([
        {"layer": "AdaptiveAvgPool2d", "shape": [channels, *model.pool.output_size], "parameters": 0},
        {"layer": "Flatten / Dropout", "shape": [model.head[2].in_features], "parameters": 0},
        {"layer": "Hidden Linear / ReLU / Dropout", "shape": [model.head[2].out_features],
         "parameters": sum(p.numel() for p in model.head[2].parameters())},
        {"layer": "Output Linear", "shape": [model.head[-1].out_features],
         "parameters": sum(p.numel() for p in model.head[-1].parameters())},
    ])
    return rows, sum(p.numel() for p in model.parameters())
