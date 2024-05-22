from datasets.cityscapes import CityScapes
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

# define the TRAIN_PATH, VAL_PATH
TRAIN_PATH = 'datasets/cityscapes/gtFine_trainvaltest/gtFine/train'
VAL_PATH = 'datasets/cityscapes/gtFine_trainvaltest/gtFine/val'

# define the transform
transform = transforms.Compose([
    transforms.Resize((1024, 512)),
])


train_dataset = CityScapes(TRAIN_PATH, TRAIN_PATH, transform=transform, target_transform=transform)
val_dataset = CityScapes(VAL_PATH, VAL_PATH, transform=transform, target_transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=2)


if __name__ == '__main__':
    for i, (image, label) in enumerate(train_loader):
        print(f'Image shape: {image.shape}, Label shape: {label.shape}')
        if i == 0:
            break
