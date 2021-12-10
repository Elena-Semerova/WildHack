from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision import datasets
from PIL import Image

class AnimalTestDataset(Dataset):
    def __init__(self, datapath, transform=None):
        super(AnimalTestDataset).__init__()
        self.datapath = datapath
        self.files = os.listdir(self.datapath)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
        
    def __getitem__(self, idx):
        filepath = os.path.join(self.datapath, self.files[idx])
        img = Image.open(filepath)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.files[idx]

    def __len__(self):
        return len(self.files)


def make_dataset(datapath, transform, is_test=False):
    if is_test:
        image_dataset = AnimalDataset(datapath, transform)
    else:
        image_dataset = datasets.ImageFolder(datapath, transform)

    return image_dataset, len(image_dataset)


def make_train_val_dataloaders(train_datapath, val_datapath, train_transform, val_transform, batch_size, num_workers):
    train_dataset, train_size = make_dataset(train_datapath, train_transform)
    val_dataset, val_size = make_dataset(val_datapath, val_transform)

    train_dataloader = DataLoader(train_dataset,
                                  sampler=ImbalancedDatasetSampler(train_dataset),
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    
    return train_dataloader, val_dataloader, train_size, val_size


def make_test_dataloader(test_datapath, test_transform, num_workers):
    test_dataset, test_size = make_dataset(test_datapath, test_transform, is_test=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=test_size,
                             shuffle=False,
                             num_workers=num_workers)
    
    return test_loader
