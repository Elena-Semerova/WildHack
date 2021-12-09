import torch
import torch.nn as nn
import torch.optim as optim
import subprocess

from torchvision import models, transforms
from torch.optim import lr_scheduler

import warnings
warnings.simplefilter("ignore", UserWarning)

from image_dataloaders import make_train_val_dataloaders, make_test_dataloader
from classifier import model_to_classifier
from train import train
from utils import saving, seed
from config import *


def main():
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }

    seed(SEED)

    train_dataloader, val_dataloader, train_size, val_size = make_train_val_dataloaders(TRAIN_DATAPATH,
                                                                                        VAL_DATAPATH,
                                                                                        data_transforms['train'],
                                                                                        data_transforms['val'],
                                                                                        BATCH_SIZE,
                                                                                        NUM_WORKERS)
    
    model = models.resnet18(pretrained=True)
    model, params = model_to_classifier(model, DEVICE, NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    model = train(model, train_dataloader, val_dataloader, train_size, val_size, criterion, optimizer, scheduler, DEVICE, NUM_EPOCHS)

    saving(model, MODEL_FILE_NAME)
    
if __name__ == '__main__':
    main()