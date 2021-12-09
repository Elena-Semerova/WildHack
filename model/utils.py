import torch
import random


def saving(model, name_model):
    torch.save(model, name_model + '.pth')
    torch.save(model.state_dict(), name_model + '_params' + '.pth')


def seed(value):
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)