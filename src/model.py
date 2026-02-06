
import torch.nn as nn
from torchvision import models
from config import *

def get_model():
    model=models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad=False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features,256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256,NUM_CLASSES)
    )
    return model.to(DEVICE)
        