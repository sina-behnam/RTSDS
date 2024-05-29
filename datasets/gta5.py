import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torch
import glob
from collections import namedtuple
import pandas as pd
import numpy as np

cityscape_color_map = {
    'unlabeled' :             [255,(  0,  0,  0)],
    'ego vehicle' :           [255,(  0,  0,  0)],
    'rectification border' :  [255,(  0,  0,  0)],
    'out of roi' :            [255,(  0,  0,  0)],
    'static' :                [255,(  0,  0,  0)],
    'dynamic' :               [255,(111, 74,  0)],
    'ground' :                [255,( 81,  0, 81)],
    'road' :                  [0,(128, 64,128)],
    'sidewalk' :              [1,(244, 35,232)],
    'parking' :               [255,(250,170,160)],
    'rail track' :            [255,(230,150,140)],
    'building' :              [  2,( 70, 70, 70)],
    'wall' :                  [  3,(102,102,156)],
    'fence' :                 [  4,(190,153,153)],
    'guard rail' :            [255,(180,165,180)],
    'bridge' :                [255,(150,100,100)],
    'tunnel' :                [255,(150,120, 90)],
    'pole' :                  [  5,(153,153,153)],
    'polegroup' :             [255,(153,153,153)],
    'traffic light' :         [  6,(250,170, 30)],
    'traffic sign' :          [  7,(220,220,  0)],
    'vegetation' :            [  8,(107,142, 35)],
    'terrain' :               [  9,(152,251,152)],
    'sky' :                   [ 10,( 70,130,180)],
    'person' :                [ 11,(220, 20, 60)],
    'rider' :                 [ 12,(255,  0,  0)],
    'car' :                   [ 13,(  0,  0,142)],
    'truck' :                 [ 14,(  0,  0, 70)],
    'bus' :                   [ 15,(  0, 60,100)],
    'caravan' :               [255,(  0,  0, 90)],
    'trailer' :               [255,(  0,  0,110)],
    'train' :                 [ 16,(  0, 80,100)],
    'motorcycle' :            [ 17,(  0,  0,230)],
    'bicycle' :               [ 18,(119, 11, 32)],
    'license plate' :         [ -1,(  0,  0,142)],
}

cityscape_color_map_df = pd.DataFrame(cityscape_color_map).T

class GTA5(Dataset):
    def __init__(self, images_path, labels_path, transformer, target_transofrmer):
        super(GTA5, self).__init__()

        self.transform = transformer
        self.target_transform = target_transofrmer

        self.images_filenames = glob.glob(os.path.join(images_path,'**.png'))
        self.labels_filenames = glob.glob(os.path.join(labels_path,'**.png'))

        self.images_dataset = self.__make_dataset__()

        # self.images_dataset['label'].apply(lambda x : self.label_driver(x[0]))

    def label_driver(self,label_path : str):

        label = read_image(label_path,mode=ImageReadMode.RGB).long()

        return self.__decode_label__(label,cityscape_color_map_df)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        #
        image = read_image(self.images_dataset.iloc[idx]['image']).float()
        #  Read label and transform it into rgb
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

    def __decode_label__(self,label : torch.Tensor,df : pd.DataFrame) -> torch.Tensor:
        result = torch.zeros(label.size()[1:], dtype=torch.long)

        # # convert to one channel
        for i in range(19):
            mask = torch.all(label == torch.tensor(df.loc[df[0] == i].iloc[0,1]).unsqueeze(1).unsqueeze(1), dim=0)
            result[mask] = i

        return result.unsqueeze(0)