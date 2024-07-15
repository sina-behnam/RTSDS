import warnings
warnings.filterwarnings("ignore")

import yaml
import argparse
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from models.deeplabv2 import deeplabv2
from models.domain_shift.adversarial.model import DomainDiscriminator,TinyDomainDiscriminator
from train import train, adversarial_train_2
from models.bisenet import build_bisenet
from torch.utils.data import DataLoader
from collections import namedtuple
from datasets.cityscapes import CityScapes
from datasets.gta5 import GTA5
from validation import val, val2, val_GTA5
from utils import IntRangeTransformer, forModel
from callbacks import Callback, WandBCallback
import numpy as np


def _augmentator_(name, config):
    if name == 'GaussianBlur':
        gaussian_blur_cfg = config.get('GaussianBlur')
        kernel_size = [int(i) for i in gaussian_blur_cfg['kernel_size'].split(',')]
        sigma = [float(i) for i in gaussian_blur_cfg['sigma'].split(',')]
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            
    elif name == 'RandomHorizontalFlip':
        random_horizontal_flip_cfg = config.get('RandomHorizontalFlip')
        return transforms.RandomHorizontalFlip(p=random_horizontal_flip_cfg['p'])
            
    elif name == 'ColorJitter':
        color_jitter_cfg = config.get('ColorJitter')
        brightness = color_jitter_cfg['brightness']
        contrast = color_jitter_cfg['contrast']
        saturation = color_jitter_cfg['saturation']
        hue = color_jitter_cfg['hue']
        return transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    return None

def augmentation_loader(config, probability: float) -> transforms.Compose:
    augmentation = config.augmentation

    _aug_list_ = []

    for key in augmentation.keys():
        aug = _augmentator_(key, config=augmentation)
        if aug is not None:
            _aug_list_.append(aug)
  

    return transforms.RandomApply(_aug_list_, p=probability)
        

def datasets_loader(config, is_augmented : bool) -> DataLoader:

    cityscapes = config.data.get('cityscapes')
    gta5 = config.data.get('gta5_modified')
    
    cityscapes_image_size = [int(i) for i in cityscapes['image_size'].split(',')]
    gta5_image_size = [int(i) for i in gta5['image_size'].split(',')]

    # Create a transforms pipeline for Cityscapes
    input_transform_Cityscapes = transforms.Compose([
        transforms.Resize(cityscapes_image_size, antialias=True), # 'antialias = True' ensures that the resizing operation uses antialiasing, which can produce better quality images when downscaling
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    output_transform_Cityscapes = transforms.Compose([
        transforms.Resize(cityscapes_image_size, antialias=True),
        IntRangeTransformer(min_val=0, max_val=cityscapes['num_classes'])
    ])

    # Create a transforms pipeline for GTA5
    
    GTA5_transforms = [
        transforms.Resize(gta5_image_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if is_augmented:
        Augmentations = augmentation_loader(config, probability=0.5)
        GTA5_transforms.insert(0, Augmentations)

    input_transform_GTA5 = transforms.Compose(GTA5_transforms)

    
    output_transform_GTA5 = transforms.Compose([
        transforms.Resize(gta5_image_size),
    ])

    # For Cityscapes
    train_datasets = CityScapes(cityscapes['segmentation_train_dir'], cityscapes['images_train_dir'], input_transform_Cityscapes, output_transform_Cityscapes)
    val_datasets = CityScapes(cityscapes['segmentation_val_dir'], cityscapes['images_val_dir'], input_transform_Cityscapes, output_transform_Cityscapes)

    train_dataloader = DataLoader(train_datasets, batch_size=cityscapes['batch_size'], shuffle=True, pin_memory=True, num_workers=cityscapes['num_workers'])
    val_dataloader = DataLoader(val_datasets, batch_size=cityscapes['batch_size'], shuffle=False, pin_memory=True, num_workers=cityscapes['num_workers'])

    # For GTA5
    train_datasetGTA5 = GTA5(gta5['images_dir'], gta5['segmentation_dir'], input_transform_GTA5, output_transform_GTA5)
    train_dataloaderGTA5 = DataLoader(train_datasetGTA5, batch_size=gta5['batch_size'], shuffle=True, pin_memory=True, num_workers=gta5['num_workers'])

    return train_dataloader, val_dataloader, train_dataloaderGTA5

def optimzer_loss_loader(model, optimizer_config: dict, loss_config: dict):
    if optimizer_config['name'] == 'Adam':
        try:
            weight_decay = optimizer_config['weight_decay']
        except KeyError:
            weight_decay = 0
        optimizer = optim.Adam(model.parameters(), lr=optimizer_config['lr'],
                               weight_decay=weight_decay)
    elif optimizer_config['name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=optimizer_config['lr'],
                               momentum=optimizer_config['momentum'])
    else:
        raise ValueError('Invalid optimizer name. Please select Adam or SGD')
    
    if loss_config['name'] == 'CrossEntropy':
        try:
            ignore_index = loss_config['ignore_index']
        except KeyError:
            ignore_index = None
        
        loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_config['name'] == 'BCEWithLogits':
        loss = nn.BCEWithLogitsLoss()
    else:
        raise ValueError('Invalid loss name. Please select CrossEntropy or BCEWithLogits')

    return optimizer, loss   

def model_loader(config : dict,is_adversarial : bool, model_name : str):

    """_summary_

    This function loads the model and the optimizer and loss function for the model.

    If the `is_adversarial` flag is set to True, 
    the function will load the generator and the discriminator models,
    along with their respective optimizers and loss functions.

    Args:
    config : dict
        The configuration file loaded using the yaml module
    is_adversarial : bool
        A flag to indicate if the adversarial training is to be performed
    model_name : str
        The name of the model to be loaded. The available models are deeplab and bisenet
    
    Returns:
    model : torch.nn.Module
        The model to be trained
    optimizer : torch.optim
        The optimizer to be used for training the model
    loss : torch.nn.Module
        The loss function to be used for training the model
    hyperparameters : dict
        The hyperparameters for the model
    """

    model_cfg = config.model

    if is_adversarial:
        adversarial_cfg = model_cfg.get('adversarial_model')

        if adversarial_cfg.get('generator')['name'] == 'bisenet':
            generator = build_bisenet.BiSeNet(num_classes=model_cfg['bisenet']['num_classes'], context_path=model_cfg['bisenet']['backbone'])

            generator_optimizer_loss = optimzer_loss_loader(generator,
                                                        adversarial_cfg['generator']['optimizer'],
                                                        adversarial_cfg['generator']['criterion']
                                                        )
            
            generator_hparams = {
                'gen_init_lr': adversarial_cfg['generator']['optimizer']['lr'],
                'gen_power': adversarial_cfg['generator']['power_lr_factor'],
            } 
            
        if adversarial_cfg.get('discriminator')['name'] == 'tiny':
            discriminator = TinyDomainDiscriminator(num_classes=adversarial_cfg['discriminator']['input_channels'])

            discriminator_optimizer_loss = optimzer_loss_loader(discriminator,
                                                        adversarial_cfg['discriminator']['optimizer'],
                                                        adversarial_cfg['discriminator']['criterion']
                                                        )
            
            discriminator_hparams = {
                'dis_init_lr': adversarial_cfg['discriminator']['optimizer']['lr'],
                'dis_power': adversarial_cfg['discriminator']['power_lr_factor'],
            }

        generator = forModel(generator, config.device)
        discriminator = forModel(discriminator, config.device)

        return (generator, generator_optimizer_loss[0], generator_optimizer_loss[1],generator_hparams), \
                (discriminator, discriminator_optimizer_loss[0], discriminator_optimizer_loss[1], discriminator_hparams)
        
    if model_name == 'deeplab':
        deeplab_cfg = model_cfg.get('deeplab')
        model = deeplabv2.get_deeplab_v2(deeplab_cfg['num_classes'], pretrain=deeplab_cfg['pretrain'], pretrain_model_path=deeplab_cfg['pretrain_model_path'])
        
        optimizer_loss = optimzer_loss_loader(model, model_cfg['deeplab']['optimizer'], model_cfg['deeplab']['criterion'])

        model_hparams = {
            'init_lr': model_cfg['deeplab']['optimizer']['lr'],
            'power': model_cfg['deeplab']['power_lr_factor']
        }

    elif model_name == 'bisenet':
        bisenet_cfg = model_cfg.get('bisenet')
        model = build_bisenet.BiSeNet(num_classes=bisenet_cfg['num_classes'], context_path=bisenet_cfg['backbone'])

        optimizer_loss = optimzer_loss_loader(model, model_cfg['bisenet']['optimizer'], model_cfg['bisenet']['criterion'])

        model_hparams = {
            'init_lr': model_cfg['bisenet']['optimizer']['lr'],
            'power': model_cfg['bisenet']['power_lr_factor']
        }

    else:
        raise ValueError('Invalid model name. Please select deeplab or bisenet')
    
    model = forModel(model, config.device)
    
    return model, optimizer_loss[0], optimizer_loss[1], model_hparams

def argumnet_parser():

    ## write a parser for the arguments
    parser = argparse.ArgumentParser(description='Semantic Segmentation and Domain Adaptation')
    ## defining the config.yaml file
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file. [Default is config.yaml.]')
    # setting the augmentation flag
    parser.add_argument('--dataset', type=str, default='cityscapes', help='Dataset to be used for training. \
                         This option only appliable to training without domain adaptation purpose \
                        [The available datasets are cityscapes and gta5. Default is cityscapes.]')

    parser.add_argument('--augmented', action='store_true', help='If augmentation is to be performed on the dataset. \
                        The augmentation is only applied on GTA5 images dataset. [Default is False.]')
    ## add to the parser if it adversarial or not
    parser.add_argument('--domain_adaptation', action='store_true', help='If Doamin Adaptation need to be performed, \
                        Currently only adversarial training is supported. \n \
                        [If this flag is set, the model will be trained using adversarial training.]')
    # adding the model 
    parser.add_argument('--model', type=str, default='bisenet', help='Model to be used for training for Generating Segmentation Maps. \
                                                                    [The available models are deeplab and bisenet. Default is bisenet.]')
    # if log_wandb is set to True
    parser.add_argument('--wandb', action='store_true', help='If the training logs need to be logged to Wandb Platform \
                        [If this flag is set, the training logs will be logged to Wandb Platform]')
    # set seed
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility. [Default is 42.]')    


    return parser.parse_args()


# def set seed for torch and if the cuda is available set the seed for cuda as well
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # set seed for numpy
    np.random.seed(seed)

if __name__ == '__main__':
    args = argumnet_parser()

    set_seed(args.seed)

    try: 
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config = namedtuple('Config', config.keys())(*config.values())
    except FileNotFoundError:
        raise FileNotFoundError('Config file not found. Please provide the correct path to the config file.')
    
    is_adversarial = False
    if args.domain_adaptation:
        is_adversarial = True

    # loading the datasets
    train_dataloader, val_dataloader, train_dataloaderGTA5 = datasets_loader(config,
                                                                              is_augmented=True if args.augmented else False)
    
    callbacks_cfg = config.callbacks
    callbacks_cfg_logging = callbacks_cfg.get('logging')
    callbacks = []
    if args.wandb:
        config_dict = config._asdict()
        callbacks.append(WandBCallback(project_name=callbacks_cfg_logging['wandb']['project_name'],
                        run_name=callbacks_cfg_logging['wandb']['run_name'],
                        config=config_dict,
                        note=callbacks_cfg_logging['wandb']['note']))
        
    # loading the model
    if is_adversarial:
        generator_complex, discriminator_complex = model_loader(config, is_adversarial=True, model_name=args.model)
        generator, generator_optimizer, generator_loss, generator_hparams = generator_complex
        discriminator, discriminator_optimizer, discriminator_loss, discriminator_hparams = discriminator_complex

        training_cfg = config.training['domain_adaptation']

        adversarial_train_2(
            iterations=training_cfg['iterations'],
            epochs=training_cfg['epochs'],
            lambda_=training_cfg['lambda'],
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            source_dataloader=train_dataloaderGTA5,
            target_dataloader=train_dataloader,
            gen_init_lr= generator_hparams['gen_init_lr'],
            dis_init_lr= discriminator_hparams['dis_init_lr'],
            lr_decay_iter= training_cfg['lr_decay_iter'],
            gen_power= generator_hparams['gen_power'],
            dis_power= discriminator_hparams['dis_power'],
            num_classes= training_cfg['num_classes'],
            class_names= config.meta['class_names'],
            val_loader=val_dataloader,
            do_validation=training_cfg['do_validation'],
            when_print=training_cfg['when_print'],
            callbacks=callbacks,
            device=config.device
        ),        

    else:

        if args.dataset == 'gta5':
            print(' ------> The Training the model on GTA5 dataset and validating on Cityscapes dataset ------ ')
            train_dataloader = train_dataloaderGTA5

        model_complex = model_loader(config, is_adversarial=False, model_name=args.model)
        model, optimizer, criterion, model_hparams = model_complex

        training_cfg = config.training.get('segmentation')

        max_iter = training_cfg['epochs'] * len(train_dataloader)

        for epoch in range(training_cfg['epochs']):

            train(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_dataloader,
                epoch=epoch,
                init_lr=model_hparams['init_lr'],
                lr_decay_iter=training_cfg['lr_decay_iter'],
                power=model_hparams['power'],
                max_iter=max_iter,
                callbacks=callbacks,
                device=config.device
            )

            val2(
                epoch=epoch,
                model=model,
                val_loader=val_dataloader,
                num_classes=training_cfg['num_classes'],
                class_names=config.meta['class_names'],
                detailed_report=True,
                device=config.device,
                callbacks=callbacks
            )
                                                   