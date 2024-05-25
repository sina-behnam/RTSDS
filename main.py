import argparse
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from models.deeplabv2 import deeplabv2
from train import train, val, val_GTA5
from models.bisenet import build_bisenet

# from fvcore.nn import FlopCountAnalysis, flop_count_table
from statistics import mean, stdev

from torch.utils.data import DataLoader


from datasets.cityscapes import CityScapes
from datasets.gta5 import GTA5
from utils import IntRangeTransformer

class_names = [
        "road", "sidewalk", "building",  "wall", "fence", "pole", "traffic light", "traffic sign",
        "vegetation", "terrain", "sky", "person", "rider", "car",
        "truck", "bus", "train", "motorcycle", "bicycle"
    ]


# Define loss and optimizer
def LossAndOptimizer(model):
    learning_rate = 1e-3
    
    criterion = nn.CrossEntropyLoss(ignore_index=19)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    return criterion, optimizer

def forModel(model,device):
    if device == 'cuda':
        model = model.cuda()  # Move model to GPU if available
        print('The number of cuda GPUs : ',torch.cuda.device_count())

    # If you have multiple GPUs available
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model) # Use DataParallel to wrap your model
        
    return model

def run_deep_lab(
                train_epochs,
                init_lr = 1e-3,
                device = 'cpu',
                train_dataloader = None,
                num_classes = 19,
                pretrain=True,
                pretrain_model_path='../data/deeplab_resnet_pretrained_imagenet.pth'):
    
    # Initialize Model
    model_DeepLab = forModel(deeplabv2.get_deeplab_v2(num_classes, pretrain=pretrain, pretrain_model_path=pretrain_model_path)
                             , device)

    criterion, optimizer = LossAndOptimizer(model_DeepLab)
    max_iter = train_epochs * len(train_dataloader)

    # Training Loop and Validation loop
    for epoch in range(train_epochs):
        # Train
        train(epoch, model_DeepLab, train_dataloader, criterion, optimizer, init_lr, max_iter)

        # Validation
        val(epoch, model_DeepLab, val_dataloader, num_classes, device)


def run_biesnet(
                train_epochs = 3,
                init_lr = 1e-3,
                device = 'cpu',
                train_dataloaderGTA5= None,
                val_dataloader= None,
                class_names = None,
                context_path='resnet18'):

    # BiseNet
    # Initialize Model
    model_BisNet_18 = forModel(build_bisenet.BiSeNet(num_classes=19, context_path=context_path),device)

    criterion, optimizer = LossAndOptimizer(model_BisNet_18)

    # Model Configuration
    train_epochs = 3 #50

    init_lr = 1e-3
    max_iter = train_epochs * len(train_dataloader)

    # Training Loop
    for epoch in range(train_epochs):

        # Training loop
        train(epoch, model_BisNet_18, train_dataloaderGTA5, criterion, optimizer, init_lr, max_iter, device='cpu')

        # Validation loop
        test_accuracy = val_GTA5(epoch, model_BisNet_18, val_dataloader, 19, class_names, 'cpu')


def argumnet_parser():
    parser = argparse.ArgumentParser(description='BiseNet and DeepLab Training')
    # model
    parser.add_argument('--model', type=str, default='bisenet', help='Model to train. Options: bisenet, deeplab')
    # device
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the training on. Options: cpu, cuda')
    # epochs
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train the model')
    # learning rate
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    # pretrain
    parser.add_argument('--pretrain', type=bool, default=True, help='Use pretrained model')
    # pretrain_model_path
    parser.add_argument('--pretrain_model_path', type=str, default='../data/deeplab_resnet_pretrained_imagenet.pth', help='Path to the pretrained model')
    # add argument for Train for Cityscapes Batch_size and image size
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--image_size', type=tuple, default=(512, 1024), help='Image size for training')
    # add argument for Train for GTA5 Batch_size = 4 and image size = 720x1280
    parser.add_argument('--batch_size_GTA5', type=int, default=4, help='Batch size for training GTA5')
    parser.add_argument('--image_size_GTA5', type=tuple, default=(720, 1280), help='Image size for training GTA5')


    return parser.parse_args()

if __name__ == '__main__':
    args = argumnet_parser()

    
    
        # defining the train, val datasets path
    TRAIN_PATH_CityScapes = ['../data/Cityscapes/Cityspaces/gtFine/train',
                  '../data/Cityscapes/Cityspaces/images/train'
                  ]

    VAL_PATH_CityScapes = ['../data/Cityscapes/Cityspaces/gtFine/val',
                  '../data/Cityscapes/Cityspaces/images/val'
                  ]

    TRAIN_PATH_GTA5 = ['../data/GTA5/images',
                  '../data/GTA5/labels'
                  ]

    num_classes = 19

    # Create a transforms pipeline for Cityscapes
    input_transform_Cityscapes = transforms.Compose([
        transforms.Resize(args.image_size, antialias=True), # 'antialias = True' ensures that the resizing operation uses antialiasing, which can produce better quality images when downscaling
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    output_transform_Cityscapes = transforms.Compose([
        transforms.Resize(args.image_size, antialias=True),
        IntRangeTransformer(min_val=0, max_val=19)
    ])

    # Create a transforms pipeline for GTA5
    input_transform_GTA5 = transforms.Compose([
        transforms.Resize(args.image_size_GTA5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    output_transform_GTA5 = transforms.Compose([
        transforms.Resize(args.image_size_GTA5),
    ])

    # For Cityscapes
    train_datasets = CityScapes(TRAIN_PATH_CityScapes[0], TRAIN_PATH_CityScapes[1], input_transform_Cityscapes, output_transform_Cityscapes)
    val_datasets = CityScapes(VAL_PATH_CityScapes[0], VAL_PATH_CityScapes[1], input_transform_Cityscapes, output_transform_Cityscapes)

    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    # For validation and testing, shuffling is typically set to False. 
    #This ensures that the validation/testing result
    # s are reproducible and the same samples are evaluated in the same order each time.
    val_dataloader = DataLoader(val_datasets, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # For GTA5
    train_datasetGTA5 = GTA5(TRAIN_PATH_GTA5[0], TRAIN_PATH_GTA5[1], input_transform_GTA5, output_transform_GTA5)
    train_dataloaderGTA5 = DataLoader(train_datasetGTA5, batch_size=args.batch_size_GTA5, shuffle=True, pin_memory=True, num_workers=2)


    if args.model == 'deeplab':
        run_deep_lab(args.epochs, args.lr, args.device, train_dataloader, num_classes, args.pretrain, args.pretrain_model_path)
    elif args.model == 'bisenet':
        run_biesnet(args.epochs, args.lr, args.device, train_dataloaderGTA5, val_dataloader, class_names, 'resnet18')
    else:
        print('Invalid model type. Please select bisenet or deeplab')