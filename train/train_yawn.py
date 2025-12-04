#python -m train.train_yawn
import time
from pathlib import Path
import torch
from torch import nn, optim
from torchvision.models import resnet18
from .dataset import make_loaders
from .config import CHECKPOINT_DIR

TASK_NAME = 'yawn'
SUBFOLDERS = ['no_yawn','yawn']
IMG_SIZE = 128
CKPT_PATH = CHECKPOINT_DIR / 'yawn_best.pt'

EPOCHS = 10
BATCH  = 32
LR     = 1e-3
WD     = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running=0.0; correct=0; total=0
    for x,y in loader:
        x=x.to(DEVICE); y=y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item()*x.size(0)
        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    return running/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running=0.0; correct=0; total=0
    for x,y in loader:
        x=x.to(DEVICE); y=y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        running += loss.item()*x.size(0)
        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    return running/total, correct/total

def main(DATASET_DIR):
    print(f"Task: {TASK_NAME}  Device: {DEVICE}")
    train_loader, val_loader, classes = make_loaders(DATASET_DIR, SUBFOLDERS, IMG_SIZE, BATCH)
    print('Classes:', classes)

    model = resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best_acc=0.0

    for epoch in range(1, EPOCHS+1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
        vl, va = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d}: train_loss={tl:.4f} train_acc={ta:.3f} | val_loss={vl:.4f} val_acc={va:.3f}")
        if va>best_acc:
            best_acc=va
            torch.save({'model': model.state_dict(), 'classes': classes}, CKPT_PATH)
            print('  -> saved', CKPT_PATH)
    print('Best val acc:', best_acc)
if __name__=='__main__':
    from .config import YAWN_DATASET_DIR
    main(YAWN_DATASET_DIR)
# train miệng 