import torch
from torch import nn
from torchvision.models import resnet18
from .dataset import make_loaders
from .config import EYE_DATASET_DIR, YAWN_DATASET_DIR, EYE_CKPT, YAWN_CKPT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_and_eval(ckpt_path, subfolders, img_size, root):
    train_loader, val_loader, classes = make_loaders(root, subfolders, img_size, 32)
    model = resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 2)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.to(DEVICE); model.eval()
    criterion = nn.CrossEntropyLoss()

    loss=0.0; correct=0; total=0
    with torch.no_grad():
        for x,y in val_loader:
            x=x.to(DEVICE); y=y.to(DEVICE)
            out=model(x); loss+=criterion(out,y).item()*x.size(0)
            pred=out.argmax(1); correct+=(pred==y).sum().item(); total+=y.size(0)
    print(ckpt_path.name, 'acc=', correct/total if total else 0.0, 'loss=', loss/total if total else 0.0)

if __name__=='__main__':
    from pathlib import Path
    if Path(EYE_CKPT).exists():
        load_and_eval(EYE_CKPT, ['Closed','Open'], 128, EYE_DATASET_DIR)
    else:
        print('Missing', EYE_CKPT)
    if Path(YAWN_CKPT).exists():
        load_and_eval(YAWN_CKPT, ['no_yawn','yawn'], 128, YAWN_DATASET_DIR)
    else:
        print('Missing', YAWN_CKPT)

