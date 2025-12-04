import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def make_transforms(img_size=128, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(0.1,0.1,0.1,0.05),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

def make_loaders(root, subnames, img_size=128, batch=32, num_workers=2):
    train_ds = datasets.ImageFolder(root / 'train', transform=make_transforms(img_size, True))
    val_ds   = datasets.ImageFolder(root / 'test',  transform=make_transforms(img_size, False))
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=num_workers),
        train_ds.classes
    )
# file xử lý ảnh trước khi nạp
