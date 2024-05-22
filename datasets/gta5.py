import os
from torch.utils.data import Dataset
from PIL import Image

class GTA5(Dataset):
    def __init__(self, root, image_folder='images', label_folder='labels', transform=None, target_transform=None):
        super(GTA5, self).__init__()
        self.image_root = os.path.join(root, image_folder)
        self.label_root = os.path.join(root, label_folder)
        self.transform = transform
        self.target_transform = target_transform
        
        # Assuming that images and labels have the same filename except for the extension
        self.images = sorted(os.listdir(self.image_root))
        self.labels = sorted(os.listdir(self.label_root))

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.images[idx])
        label_path = os.path.join(self.label_root, self.labels[idx])

        # Load image and label
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.images)