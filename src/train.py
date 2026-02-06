import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_dataloaders
from model import get_model
from config import *

def train_epoch(model,loader,criterion,optimizer):
    model.train()
    total_loss=0

    for imgs,labels in tqdm(loader):
        imgs,labels = imgs.to(DEVICE),labels.to(DEVICE)

        optimizer.zero_grad()
        outputs=model(imgs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss=loss.item()

    return total_loss / len(loader)

def validate(model,loader,criterion):
    model.eval()
    correct=0
    total=0
    loss_sum=0
    with torch.no_grad():
        for imgs,labels in tqdm(loader):
            imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
            outputs=model(imgs)
            loss=criterion(outputs,labels)
            _,pred=torch.max(outputs,1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            loss_sum += loss.item()

    return loss_sum / len(loader),correct / total


def main():
    train_loader , val_loader , classes=get_dataloaders()
    print("Classes",classes)

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    best_acc=0

    for epoch in range(EPOCHS):
        train_loss=train_epoch(model,train_loader,criterion,optimizer)
        val_loss,val_acc=validate(model,val_loader,criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss} | "
              f"Val Acc: {val_acc}")

        if val_acc > best_acc:
            best_acc=val_acc
            torch.save(model.state_dict(),"best_model.pth")

if __name__ == "__main__":
    main()