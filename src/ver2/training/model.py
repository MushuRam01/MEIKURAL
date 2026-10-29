"""Version 2 deliberately uses the exact Version 1 CNN and device selection."""

from ...ver1.training.model import RaagCNN, model_dimensions, select_device

__all__ = ['RaagCNN', 'model_dimensions', 'select_device']
