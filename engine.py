import torch
import utils

def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: torch.nn.Module,
               device: torch.device):

    running_loss = 0.
    last_loss = 0.

    model.train()

    for i, (lr_images, hr_images) in enumerate(train_dataloader):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        optimizer.zero_grad()

        outputs = model(lr_images)

        loss = criterion(outputs, hr_images)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10
            print('batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss

def test_step(model: torch.nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              device: torch.device):

    running_vloss = 0.0

    model.eval()

    with torch.no_grad():
        for i, (vinputs,vlabels) in enumerate(test_dataloader):
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)

    return avg_vloss

def train(EPOCHS: int,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          model_path: str,
          device: torch.device):

    best_vloss = 1000000

    for epoch in range(1,EPOCHS+1):
        print('EPOCH {}:'.format(epoch))

        avg_loss = train_step(model, train_dataloader, optimizer, criterion, device)
        avg_vloss = test_step(model, train_dataloader, criterion, device)

        print('LOSS train {} test {}'.format(avg_loss, avg_vloss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            utils.save_model(model,  model_path, epoch, optimizer, best_vloss)
