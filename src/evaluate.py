import torch
from sklearn.metrics import classification_report , confusion_matrix
from dataset import get_dataloaders
from model import get_model
from config import *

_,val_loader,class_name=get_dataloaders()
model=get_model()
model.load_state_dict(torch.load("model/best_model.pth"))
model.eval()

y_true,y_pred=[],[]

with torch.no_grad():
    for imgs,labels in val_loader:
        imgs=imgs.to(DEVICE)
        outputs=model(imgs)
        _,pred=torch.max(outputs,1)

        y_true.extend(labels.numpy())
        y_pred.extend(pred.cpu().numpy())

print(confusion_matrix(y_true,y_pred))
print(classification_report(y_true, y_pred, target_names=class_name))