import numpy as np
import matplotlib.pyplot as plt


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.1)


def visualize_predictions(model, dataloader, class_names, num_images=8):
    inputs, classes = next(iter(dataloader))
    classes_val = torch.argmax(model(inputs), dim=-1)
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes_val])