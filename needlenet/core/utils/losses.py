import torch
import torch.nn as nn

def get_loss(cfg, class_weights=None):
    loss_name = cfg['training']['loss'].lower()
    if loss_name == "cross_entropy":
        label_smoothing = cfg['training'].get('label_smoothing', 0.0)
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")