import torch

DEVICE= "cuda" if torch.cuda.is_available() else "cpu"

#Image
IMG_SIZE=224
BATCH_SIZE=32

#Traning
EPOCHS=10
LR=1e-4
NUM_CLASSES=3

#Dataset Path
DATA_DIR="data/Lung X-Ray Image/Lung X-Ray Image"