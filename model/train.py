import torch
import time

def train(model, train_dataloader, val_dataloader, train_size, val_size, criterion, optimizer, scheduler, device, num_epochs=25, mode=None):
    start = time.time()

    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        start_epoch = time.time()
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 50)

        train_epoch(model, train_dataloader, train_size, criterion, optimizer, scheduler, device)
        val_acc = val_epoch(model, val_dataloader, val_size, criterion, optimizer, device)

        if val_acc >= best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
        
        time_epoch = time.time() - start_epoch
        print('Time for epoch {:.0f}m {:.0f}s\n'.format(time_epoch // 60, time_epoch % 60))

    time_training = time.time() - start

    print('Training complete in {:.0f}m {:.0f}s'.format(time_training // 60, time_training % 60))
    print('Best val acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    if mode == 'acc':
        return model, best_acc
    else:
        return model


def train_epoch(model, dataloader, dataset_size, criterion, optimizer, scheduler, device):
    model.train()
    scheduler.step()

    train_loss = 0.0
    train_acc = 0

    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        train_acc += torch.sum(preds == labels).type(torch.float)

    epoch_loss = train_loss / dataset_size
    epoch_acc = train_acc / dataset_size

    print('\t train loss: {:.4f} \t train acc: {:.4f}'.format(epoch_loss, epoch_acc))

def val_epoch(model, dataloader, dataset_size, criterion, optimizer, device):
    model.train(False)

    val_loss = 0.0
    val_acc = 0

    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)

        val_loss += loss.item()
        val_acc += torch.sum(preds == labels).type(torch.float)

    epoch_loss = val_loss / dataset_size
    epoch_acc = val_acc / dataset_size

    print('\t val loss: {:.4f} \t val acc: {:.4f}'.format(epoch_loss, epoch_acc))

    return epoch_acc
