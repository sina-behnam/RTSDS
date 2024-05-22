from torch.utils.data import Dataset
import torch
import glob
import os
import pandas as pd
from torchvision.io import read_image
from collections import namedtuple
import sys

sys.path.append('datasets')

class CityScapes(Dataset):
    def __init__(self, annotation_path: str, images_path: str, transform=None, target_transform=None):
        super(CityScapes, self).__init__()

        images_path = self.__check_path__(images_path)
        annotation_path = self.__check_path__(annotation_path)

        self.images_filename = glob.glob(os.path.join(images_path, '**', '*.png'), recursive=True)
        self.annotations_filename = glob.glob(os.path.join(annotation_path, '**', '*.png'), recursive=True)

        # Check the ID
        self.image_dataset = self.__merge_ids__()

        self.transform = transform
        self.target_transform = target_transform

    def __check_path__(self, path: str) -> str:
        return path.rstrip('/\\')

    def __merge_ids__(self):
        def get_id(path: str) -> str:
            return '_'.join(path.split('/')[-1].split('_')[:3])

        is_colored_annotation = lambda s: s.endswith('color.png')

        img_set = {}
        Image = namedtuple('Image', ['path', 'labels'])
        for image in self.images_filename:
            l = ['\0', '\0']
            img_set[get_id(image)] = Image(image, l)

        for label in self.annotations_filename:
            id = get_id(label)
            if is_colored_annotation(label):
                img_set[id].labels[1] = label
            else:
                img_set[id].labels[0] = label

        return pd.DataFrame(list(img_set.values()))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = read_image(self.image_dataset.iloc[idx]['path']).float()  # Convert to float
        label = read_image(self.image_dataset.iloc[idx]['labels'][0]).long()  # Assuming labels are integers

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.image_dataset)