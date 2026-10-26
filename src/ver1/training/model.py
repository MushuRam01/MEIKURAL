"""Small spectrogram CNN and device selection shared with inference."""

import torch
from torch import nn

from ..config import feature_shape


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use device=auto/cpu or install compatible drivers and CUDA PyTorch")
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(requested)


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
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(m["dropout"]),
                                  nn.Linear(incoming * m["adaptive_pool"][0] * m["adaptive_pool"][1],
                                            len(config["classes"])))

    def forward(self, matrix):
        return self.head(self.pool(self.features(matrix)))


def model_dimensions(config):
    """Derive layer shapes, parameters, and receptive fields from real modules."""
    model = RaagCNN(config)
    x = torch.zeros(1, 1, *feature_shape(config))
    rows = [{"layer": "Input", "shape": list(x.shape[1:]), "parameters": 0,
             "receptive_field": [1, 1]}]
    receptive, jump = [1, 1], [1, 1]
    with torch.no_grad():
        for index, layer in enumerate(model.features):
            x = layer(x)
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                kernel = layer.kernel_size
                stride = layer.stride
                if isinstance(stride, int):
                    stride = (stride, stride)
                for axis in range(2):
                    receptive[axis] += (kernel[axis] - 1) * jump[axis]
                    jump[axis] *= stride[axis]
                rows.append({"layer": f"{type(layer).__name__} {index // 3 + 1}",
                             "shape": list(x.shape[1:]),
                             "parameters": sum(p.numel() for p in layer.parameters()),
                             "receptive_field": receptive.copy()})
        x = model.pool(x)
        rows.append({"layer": "AdaptiveAvgPool2d", "shape": list(x.shape[1:]), "parameters": 0})
        x = model.head[0](x)
        rows.append({"layer": "Flatten / Dropout", "shape": list(x.shape[1:]), "parameters": 0})
        x = model.head[2](x)
        rows.append({"layer": "Linear", "shape": list(x.shape[1:]),
                     "parameters": sum(p.numel() for p in model.head[2].parameters())})
    return rows, sum(p.numel() for p in model.parameters())
