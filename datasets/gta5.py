import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import glob
from collections import namedtuple
import pandas as pd

class GTA5(Dataset):
    def __init__(self, images_path, labels_path, transformer, target_transofrmer):
        super(GTA5, self).__init__()

        self.transform = transformer
        self.target_transform = target_transofrmer

        self.images_filenames = glob.glob(os.path.join(images_path,'**.png'))
        self.labels_filenames = glob.glob(os.path.join(labels_path,'**.png'))

        self.images_dataset = self.__make_dataset__()


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = read_image(self.images_dataset.iloc[idx]['image']).float()
        label = read_image(self.images_dataset.iloc[idx]['label'][0]).long()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image,label

    def __make_dataset__(self):

      get_id = lambda path : '_'.join(path.split('/')[-1].split('.')[0])

      img_set = {}
      Image = namedtuple('Image',['image','label'])
      for image in self.images_filenames:
        l = ['\0']
        img_set[get_id(image)] = Image(image,l)

      for label in self.labels_filenames:
        id = get_id(label)
        img_set[id].label[0] = label

      return pd.DataFrame(list(img_set.values()))


    def __len__(self):
        return len(self.images_filenames)
