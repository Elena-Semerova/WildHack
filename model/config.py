import torch

TRAIN_DATAPATH = 'dataset_animals/train'
VAL_DATAPATH = 'dataset_animals/val'

BATCH_SIZE = 4
NUM_WORKERS = 2
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
MOMENTUM = 0.9
STEP_SIZE = 7
GAMMA = 0.1

CLASS_NAMES = ['another', 'bear', 'deer', 'fox', 'hog', 'lynx',
               'saiga', 'steppe eagle', 'tiger', 'wolf']
NUM_CLASSES = len(CLASS_NAMES)

SEED = 3407

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_FILE_NAME = 'resnet18_' + str(NUM_EPOCHS) + '_cat_' + str(NUM_CLASSES)