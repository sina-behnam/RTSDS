# New train function, more correct
import numpy as np
import torch
import utils
from callbacks import Callback
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F


def train(
    epoch : int,
    model : torch.nn.Module,
    train_loader : DataLoader,
    criterion : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    init_lr : float,
    max_iter : int,
    power : float =0.9,
    lr_decay_iter : float =1.0,
    device : str ='cpu',
    callbacks : list[Callback]  = []):
    '''Training function for a single epoch.

    Args:
        epoch (int): Current epoch number
        model (nn.Module): Model to train
        train_loader (DataLoader): Training loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        init_lr (float): Initial learning rate
        max_iter (int): Maximum number of iterations
        power (float): Power factor for polynomial learning rate decay
        lr_decay_iter (flaot): Learning rate decay interval
        device (str): Device to run the training on
        callbacks (list): List of callback functions to run during training
    
    Returns:    
        nn.Module: Trained model
    '''

    # setup callbacks
    for callback in callbacks:
        callback.on_train_begin()

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    try:
        from IPython import get_ipython
        if get_ipython():
            from tqdm.notebook import tqdm
    except:
        from tqdm import tqdm
    
    # Iterate over the data
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}', leave=False):
        current_iter = epoch * len(train_loader) + batch_idx  # Calculate global iteration count across all epochs
        # Update learning rate
        if current_iter % lr_decay_iter == 0 and current_iter <= max_iter:
            utils.poly_lr_scheduler(optimizer, init_lr, current_iter, lr_decay_iter, max_iter, power)

        inputs = inputs.to(device)
        targets = targets.to(device).squeeze(1)  # Ensure targets are correctly shaped

        optimizer.zero_grad()

        # Forward pass - unpack main output and auxiliary outputs
        outputs = model(inputs)

        # If model returns a tuple, unpack it
        if isinstance(outputs, tuple):
            main_output, aux1, aux2 = outputs
        else:
            main_output, aux1, aux2 = outputs, None, None
        
        # Compute loss for the main output
        loss = criterion(main_output, targets)

        # Compute auxiliary losses if available
        if aux1 is not None:
            loss += criterion(aux1, targets)
        if aux2 is not None:
            loss += criterion(aux2, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate running loss
        running_loss += loss.item()
        
        # Compute predictions
        _, predicted = main_output.max(1)
        
        # Calculate total pixels and correctly predicted pixels for accuracy
        total += targets.size(0) * targets.size(1) * targets.size(2)  # Total number of pixels
        correct += predicted.eq(targets).sum().item()  # Sum of correctly predicted pixels

        # Run batch end callbacks
        for callback in callbacks:
            callback.on_batch_end(batch_idx, running_loss, correct, total)

    # Calculate average loss and accuracy for the epoch
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch + 1} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

    # Run epoch end callbacks
    for callback in callbacks:
        callback.on_epoch_end(epoch, train_loss, train_accuracy)

    return model


def val(
    epoch : int,
    model : torch.nn.Module,
    val_loader : DataLoader,
    num_classes : int,
    device : str = 'cpu',
    callbacks : list[Callback] = []):

    '''Validation function 

    Args:
        epoch (int): Current epoch number
        model (nn.Module): Model to evaluate
        val_loader (DataLoader): Validation loader
        num_classes (int): Number of classes in the dataset
        device (str): Device to run the evaluation on
        callbacks (list): List of callback functions to run during validation
    
    Returns:    
        float: Mean IoU for the validation set
    '''

    # Run validation begin callbacks
    for callback in callbacks:
        callback.on_validation_begin()

    model.eval()
    total_hist = np.zeros((num_classes, num_classes))  # Initialize total histogram for all classes
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze(1)  # targets should be [batch_size, height, width]

            outputs = model(inputs)

            # Verify the shape of the outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            predicted = torch.argmax(outputs, dim=1)

            # Calculate fast histogram and accumulate
            hist = utils.fast_hist(targets.cpu().numpy(), predicted.cpu().numpy(), num_classes)
            total_hist += hist

            # calculate True Positive on the confusion matrix
            TP = np.diag(total_hist)
            # calculate the Accuracy
            pixel_acc = np.sum(TP) / np.sum(total_hist)
            # loss
            loss = 1. - pixel_acc

            # Run validation end callbacks
            for callback in callbacks:
                callback.on_validation_batch_end(batch_idx, loss)

    # Compute per-class IoU from the accumulated histogram
    ious = utils.per_class_iou(total_hist)
    mean_iou = np.nanmean(ious)  # Mean IoU for reporting
    print(f'Validation Mean IoU for Epoch {epoch + 1}: {mean_iou:.4f}')

    # Run validation end callbacks
    for callback in callbacks:
        callback.on_validation_end(mean_iou)

    return mean_iou

def val_GTA5(
        epoch : int,
        model : torch.nn.Module,
        val_loader : DataLoader,
        num_classes : int,
        class_names : list[str],
        callbacks : list[Callback] = [],
        device : str = 'cpu'):
    '''Validation function for GTA5 dataset, using the provided class names for reporting IoU per class.

    Args:
        epoch (int): Current epoch number
        model (nn.Module): Model to evaluate
        val_loader (DataLoader): Validation loader
        num_classes (int): Number of classes in the dataset
        class_names (list): List of class names for reporting
        callbacks (list): List of callback functions to run during validation
        device (str): Device to run the evaluation on, default is 'cpu'
    
    Returns:
        float: Mean IoU for the validation set
    '''
    model.eval()
    total_miou = 0

    # Run validation begin callbacks
    for callback in callbacks:
        callback.on_validation_begin()

    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze(1)  # Ensure targets are [batch_size, height, width]

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Select the main output if model returns a tuple

            predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = targets.cpu().numpy()

            # Update confusion matrix using fast_hist
            confusion_matrix += utils.fast_hist(targets, predicted, num_classes)

            # calculate True Positive on the confusion matrix
            TP = np.diag(confusion_matrix)
            # calculate the Accuracy
            pixel_acc = np.sum(TP) / np.sum(confusion_matrix)
            # loss
            loss = 1. - pixel_acc
            # Run validation end callbacks
            for callback in callbacks:
                callback.on_validation_batch_end(batch_idx, loss)

    # Calculate per class IoU from the confusion matrix
    IoUs = utils.per_class_iou(confusion_matrix)
    total_miou = np.nanmean(IoUs)  # Calculate mean IoU, ignoring NaNs
    print(f'Validation mIoU for Epoch {epoch + 1}: {total_miou:.4f}')
    
    class_result_df = pd.DataFrame({'Class': class_names, 'IoU': [f'{iou:.4f}' for iou in IoUs]})
    print(class_result_df)
    # Run validation end callbacks
    for callback in callbacks:
        callback.on_validation_end({
            'validation_mIoU': total_miou
        },data=class_result_df)

    return total_miou, class_result_df

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, init_lr, iterations, power=0.9):
    lr = lr_poly(init_lr, i_iter, iterations, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adversarial_train(iterations : int ,epoch : int, generator : torch.nn.Module, discriminator : torch.nn.Module,
           generator_optimizer : torch.optim.Optimizer, discriminator_optimizer : torch.optim.Optimizer,
            source_dataloader : DataLoader, target_dataloader : DataLoader,
            generator_loss : torch.nn.Module, discriminator_loss : torch.nn.Module, lambda_ : float,
            gen_init_lr : float, power : float, dis_init_lr : float,
            device : str = 'cpu', when_print : int = 10, callbacks : list[Callback]  = []):
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
    lambda_ : float
        Lambda value for the total loss (in which is equal = generator_loss + `lambda_` * discriminator_loss)
    device : torch.device
        Device to run the model
    when_print : int
        On which iteration to print the loss values (default is 10). it should be less than the iterations
    callbacks : list
        List of callback functions to run during training
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

    # setup callbacks
    for callback in callbacks:
        callback.on_train_begin()


    running_generator_source_loss = 0.0
    running_adversarial_loss = 0.0
    running_discriminator_source_loss = 0.0
    running_discriminator_target_loss = 0.0

    generator.train()
    discriminator.train()

    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    adjust_learning_rate(generator_optimizer, epoch, gen_init_lr, iterations, power)
    adjust_learning_rate(discriminator_optimizer, epoch, dis_init_lr, iterations, power)

    for i in tqdm(range(iterations),total=iterations,desc=f'Epoch {epoch}'):

        # defining source and target data
        source_image, source_label = next(iter(source_dataloader))
        target_image, _ = next(iter(target_dataloader))

        source_image, source_label = source_image.to(device), source_label.to(device)
        source_label = source_label.squeeze(1) # removing the channel dimension
        target_image = target_image.to(device)

        # ! The Discriminator weight should not be updated during the generator training !
        for param in discriminator.parameters():
            param.requires_grad = False
        
        # ! //////////////////////// -------------  Training the Generator  ------------- //////////////////////// !
        ## * Train with the source data
        # Forward pass Generator
        # * The source features in here are same as the segmentation output as the low-dimenssion segmentation has been used as input of discriminator 
        source_gen_output = generator(source_image)
        # segmentation loss for generator
        if isinstance(source_gen_output,tuple):
            loss_gen_source = generator_loss(source_gen_output[0], source_label)
            loss_gen_source += generator_loss(source_gen_output[1], source_label)
            loss_gen_source += generator_loss(source_gen_output[2], source_label)
            source_features, _ , _ = source_gen_output
        else:
            loss_gen_source = generator_loss(source_gen_output, source_label)
            source_features = source_gen_output

        running_generator_source_loss += loss_gen_source.item()
        # backward the generator loss        
        loss_gen_source.backward()

        ## * Train with the target data
        
        target_output = generator(target_image)
        if isinstance(target_output,tuple):
            target_feature, _ , _ = target_output
        else:
            target_feature = target_output
        
        # Forward pass Discriminator
        predicted_target_domain = discriminator(F.softmax(target_feature, dim=1))

        # ! Adversarial loss 
        source_mask = torch.ones(predicted_target_domain.size()).to(device)
        loss_adversarial = lambda_ * discriminator_loss(predicted_target_domain, source_mask)

        running_adversarial_loss += loss_adversarial.item()

        loss_adversarial.backward()
        
        # ! //////////////////////// -------------  Training the Discriminator  ------------- //////////////////////// !
        ## * The Discriminator weight now should be updated during the discriminator training !
        for param in discriminator.parameters():
            param.requires_grad = True

        # detaching the features to avoid the gradient flow to the generator
        source_features = source_features.detach()
        target_feature = target_feature.detach()
        ## * Train with the source data
        predicted_source_domain = discriminator(F.softmax(source_features))
        source_mask = torch.ones(predicted_source_domain.size()).to(device)
        loss_disc_source = discriminator_loss(predicted_source_domain, source_mask)

        running_discriminator_source_loss += loss_disc_source.item()

        loss_disc_source.backward()

        ## * Train with the target data
        predicted_target_domain = discriminator(F.softmax(target_feature))
        target_mask = torch.zeros(predicted_target_domain.size()).to(device)
        loss_disc_target = discriminator_loss(predicted_target_domain, target_mask)

        running_discriminator_target_loss += loss_disc_target.item()

        loss_disc_target.backward()


        ## * ---------------------- Loggings ---------------------- * ##
        for callback in callbacks:
            callback.on_batch_end(i, {
                'loss_gen_source': loss_gen_source.item(),
                'loss_adversarial': loss_adversarial.item(),
                'loss_disc_source': loss_disc_source.item(),
                'loss_disc_target': loss_disc_target.item()
            })
        
        if when_print != -1 and (i % when_print == 0 and i != 0):
            print(f'Iteration {i}')
            utils.tabular_print({
                'loss_gen_source': running_generator_source_loss/iterations,
                'loss_adversarial': running_adversarial_loss/iterations,
                'loss_disc_source': running_discriminator_source_loss/iterations,
                'loss_disc_target': running_discriminator_target_loss/iterations
            })


    generator_optimizer.step()
    discriminator_optimizer.step()

    
