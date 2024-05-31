import torch
import warnings
from torchvision import transforms
from torch.utils.data import DataLoader

warnings.filterwarnings(action='ignore')

import datasets.cityscapes as cityscapes
import datasets.gta5 as gta5
import utils
import os


root_directory = os.path.dirname(os.path.abspath(__file__))  # Root directory of the project
datasets_directory = os.path.join(root_directory, 'data')

# Defining the train, val datasets paths for Cityscapes and GTA5
TRAIN_PATH_CityScapes = [os.path.join(datasets_directory, 'Cityscapes', 'Cityspaces', 'gtFine', 'train')
                , os.path.join(datasets_directory, 'Cityscapes', 'Cityspaces', 'images', 'train')
              ]

VAL_PATH_CityScapes = [os.path.join(datasets_directory, 'Cityscapes', 'Cityspaces', 'gtFine', 'val'),
              os.path.join(datasets_directory, 'Cityscapes', 'Cityspaces', 'images', 'val')
              ]

TRAIN_PATH_GTA5 = [os.path.join(datasets_directory, 'GTA5_Modified', 'images'),
              os.path.join(datasets_directory, 'GTA5_Modified', 'labels'),
              ]

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 19

# Create a transforms pipeline for Cityscapes
input_transform_Cityscapes = transforms.Compose([
    transforms.Resize((512, 1024), antialias=True), # 'antialias = True' ensures that the resizing operation uses antialiasing, which can produce better quality images when downscaling
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

output_transform_Cityscapes = transforms.Compose([
    transforms.Resize((512, 1024), antialias=True),
    utils.IntRangeTransformer(min_val=0, max_val=19)
])

# Create a transforms pipeline for GTA5
input_transform_GTA5 = transforms.Compose([
    transforms.Resize((720, 1280)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

output_transform_GTA5 = transforms.Compose([
    transforms.Resize((720, 1280)),

])

# For Cityscapes
train_datasets = cityscapes.CityScapes(TRAIN_PATH_CityScapes[0], TRAIN_PATH_CityScapes[1], input_transform_Cityscapes, output_transform_Cityscapes)
val_datasets = cityscapes.CityScapes(VAL_PATH_CityScapes[0], VAL_PATH_CityScapes[1], input_transform_Cityscapes, output_transform_Cityscapes)


train_dataloader = DataLoader(train_datasets, batch_size=4,shuffle=True, pin_memory=True, num_workers=2)
# For validation and testing, shuffling is typically set to False. 
#This ensures that the validation/testing results are reproducible and the same samples are evaluated in the same order each time.
val_dataloader = DataLoader(val_datasets, batch_size=4, shuffle=False, pin_memory=True, num_workers=2)

# For GTA5
train_datasetGTA5 = gta5.GTA5(TRAIN_PATH_GTA5[0], TRAIN_PATH_GTA5[1], input_transform_GTA5, output_transform_GTA5)
train_dataloaderGTA5 = DataLoader(train_datasetGTA5, batch_size=4, shuffle=True, pin_memory=True, num_workers=2)



if __name__ == '__main__':
    print('Cityscapes dataset')
    for i, data in enumerate(train_dataloader):
        print(data[0].shape, data[1].shape)
        if i > 3:
            break

    print('GTA5 dataset')
    for i, data in enumerate(train_dataloaderGTA5):
        print(data[0].shape, data[1].shape)
        if i > 3:
            break