
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np


def train(iterations : int ,epoch : int, generator : torch.nn.Module, discriminator : torch.nn.Module,
           generator_optimizer : torch.optim.Optimizer, discriminator_optimizer : torch.optim.Optimizer,
            source_dataloader : DataLoader, target_dataloader : DataLoader,
            generator_loss : torch.nn.Module, discriminator_loss : torch.nn.Module, 
            discriminator_interpolator : torch.nn.Module, image_inter_size : tuple,
            device : torch.device):
    '''
    Function to train the generator and discriminator for the adversarial training of the domain shift problem.

    Parameters:
    -----------
    iterations : int
        Number of iterations to train the model
    epoch : int
        Current epoch number
    generator : torch.nn.Module
        Generator model
    discriminator : torch.nn.Module
        Discriminator model
    generator_optimizer : torch.optim.Optimizer
        Optimizer for the generator
    discriminator_optimizer : torch.optim.Optimizer
        Optimizer for the discriminator
    source_dataloader : DataLoader
        Dataloader for the source dataset
    target_dataloader : DataLoader
        Dataloader for the target dataset
    generator_loss : torch.nn.Module
        Loss function for the generator
    discriminator_loss : torch.nn.Module
        Loss function for the discriminator
    discriminator_interpolator : torch.nn.Module
        Interpolator for the discriminator
    image_inter_size : tuple
        Size of the image after interpolation
    device : torch.device
        Device to run the model
    
    Returns:
    --------
    None
    '''

    try:
        from IPython import get_ipython
        if get_ipython():
            from tqdm.notebook import tqdm
    except:
        from tqdm import tqdm


    generator.train()
    discriminator.train()

    running_loss_gen = 0.0
    running_loss_disc = 0.0

    for i in tqdm(range(iterations),total=iterations,desc=f'Epoch {epoch}'):

        # defining source and target data
        source_image, source_label = next(iter(source_dataloader))
        target_image, _ = next(iter(target_dataloader))

        source_image, source_label = source_image.to(device), source_label.to(device)
        source_label = source_label.squeeze(1) # removing the channel dimension
        target_image = target_image.to(device)

        # * Defining the labels for discriminator as Target_DIS = 0 and the Source_DIS = 1
        batch_size_source = source_image.size(0)
        batch_size_target = target_image.size(0)
        source_mask_label = torch.ones(batch_size_source, 1, *image_inter_size).to(device)
        target_mask_label = torch.zeros(batch_size_target, 1, *image_inter_size).to(device)

        # ! Training the Discriminator
        discriminator_optimizer.zero_grad()

        # Forward pass Generator
        # * The source features in here are same as the segmentation output as the low-dimenssion segmentation has been used as input of discriminator 
        source_gen_output = generator(source_image)
        target_gen_output = generator(target_image)

        # segmentation loss for generator
        if isinstance(source_gen_output,tuple):
            gen_source_loss = generator_loss(source_gen_output[0], source_label)
            gen_source_loss += generator_loss(source_gen_output[1], source_label)
            gen_source_loss += generator_loss(source_gen_output[2], source_label)

            source_features, _ , _ = source_gen_output

        else:
            gen_source_loss = generator_loss(source_gen_output, source_label)
        
        
        if isinstance(target_gen_output,tuple):
            target_feature, _ , _ = target_gen_output
        
        # Forward pass Discriminator
        # * Here we feed the Discriminator with the output of the generator (features) or in this case the (low-dimenssion segmentation)
        source_discriminator_output = discriminator_interpolator(discriminator(F.softmax(source_features.detach())))
        target_discriminator_output = discriminator_interpolator(discriminator(F.softmax(target_feature.detach())))
        
        # loss on discriminator
        loss_disc_source = discriminator_loss(source_discriminator_output, source_mask_label)
        loss_disc_target = discriminator_loss(target_discriminator_output, target_mask_label)
        loss_disc = (loss_disc_source + loss_disc_target) / 2
        loss_disc.backward()
        discriminator_optimizer.step()
        # for logging
        running_loss_disc += loss_disc.item()
        
        # ! Train the Generator
        generator_optimizer.zero_grad()
        
        # * Adversarial loss for generator
        disc_target_preds_gen = discriminator_interpolator(discriminator(F.softmax(target_feature)))
        #
        # * Adversarial loss for generator by using the discriminator output of the target features \
        # * And the source mask label as the target label to fool the discriminator \
        # * To predict the target features as the source features.
        #
        loss_adv_gen = discriminator_loss(disc_target_preds_gen, source_mask_label)
        
        # Total generator loss
        loss_gen = gen_source_loss + loss_adv_gen
        loss_gen.backward()
        generator_optimizer.step()
        
        running_loss_gen += loss_gen.item()

        if i % 100 == 0 and i != 0:
            print(f'Iteration {i}', f"Generator Loss: {running_loss_gen/iterations:.4f}, "
              f"Discriminator Loss: {running_loss_disc/iterations:.4f}")



# past_archived
# def train(epoch : int,lambda_=0.1):

#     try:
#         from IPython import get_ipython
#         if get_ipython():
#             from tqdm.notebook import tqdm
#     except:
#         from tqdm import tqdm


#     generator.train()
#     discriminator.train()
#     for i, (source_data, target_data) in tqdm(enumerate(zip(train_dataloaderGTA5, train_dataloader)), total=len(train_dataloaderGTA5) , desc=f'Epoch {epoch}'):
#         source_image, source_label = source_data
#         target_image, _ = target_data

#         source_image, source_label = source_image.to(device), source_label.to(device)
#         target_image = target_image.to(device)

#         # ! Training the generator
#         generator_optimizer.zero_grad()
#         discriminator_optimizer.zero_grad()

#         # Forward pass Generator
#         # * The source features in here are same as the segmentation output as the low-dimenssion segmentation has been used as input of discriminator 
#         source_features = generator(source_image)
#         target_feature = generator(target_image)

#         # loss on generator of source domain 
#         # * We only perform the loss on the source domain as the target domain is not labeled
#         gen_source_loss = generator_loss_calculator(generator_loss, source_features, source_label)

#         if isinstance(source_features,tuple):
#             source_features, _ , _ = source_features
#         if isinstance(target_feature,tuple):
#             target_feature, _ , _ = target_feature
        
#         # ! Forward pass Discriminator
#         # * Here we feed the Discriminator with the output of the generator (features) or in this case the (low-dimenssion segmentation)
#         source_discriminator_output = source_interp(discriminator(F.softmax(source_features)))
#         target_discriminator_output = target_interp(discriminator(F.softmax(target_feature)))
#         # * defining the Target label as 0 and the Source label as 1
#         source_label = torch.ones_like(source_discriminator_output)
#         target_label = torch.zeros_like(target_discriminator_output)

#         # loss on discriminator
#         disc_loss = discriminator_loss(source_discriminator_output, source_label) + discriminator_loss(target_discriminator_output, target_label)
        
#         # ! Adversarial Training
#         target_feature, _, _ = generator(target_image)
#         target_discriminator_output = target_interp(discriminator(F.softmax(target_feature)))
#         # * To fool the discriminator
#         adver_loss = discriminator_loss(target_discriminator_output, source_label)
#         # total loss
#         total_loss = gen_source_loss + lambda_ * ( disc_loss + adver_loss )
#         total_loss.backward()
#         # Update the weights
#         generator_optimizer.step()
#         discriminator_optimizer.step()

#         if i % 100 == 0 and i != 0:
#             print(f'Iteration {i}, Generator Loss: {gen_source_loss.item()}, Discriminator Loss: {disc_loss.item()} , Adversarial Loss: {adver_loss.item()} , Total Loss: {total_loss.item()}')

        