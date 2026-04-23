import torch
import torchvision

    
def build_model(cfg, num_classes):
    weights = "DEFAULT" if cfg['model']['pretrained'] else None
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
    return model