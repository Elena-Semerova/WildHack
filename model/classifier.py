import torch
import torch.nn as nn
import torchvision


def model_to_classifier(model, device, num_classes):
    last_layer = nn.Linear(model.fc.in_features, num_classes, True)
    model.fc = last_layer

    activation = nn.Softmax(dim=-1)
    model = torch.nn.Sequential(*[model, activation])

    model.to(device)

    params_to_train = list(last_layer.parameters())

    return model, params_to_train