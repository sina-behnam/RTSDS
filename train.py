# New train function, more correct
import numpy as np
import torch
import utils
from callbacks import Callback
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F

try:
    from IPython import get_ipython
    
    if get_ipython():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

except:
    from tqdm import tqdm


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
            callback.on_batch_end(batch_idx, {
                'train_loss': loss.item(),
                'train_accuracy': 100. * correct / total,
            })


    # Calculate average loss and accuracy for the epoch
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch + 1} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

    # Run epoch end callbacks
    for callback in callbacks:
        callback.on_epoch_end(epoch, {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy
        })

    return model

def val2(
    epoch: int,
    model: torch.nn.Module,
    val_loader: DataLoader,
    num_classes: int,
    class_names: list[str] = None,
    detailed_report: bool = False,
    callbacks: list[Callback] = [],
    device: str = 'cpu'
):
    '''Combined validation function with optional detailed class reporting.

    Args:
        epoch (int): Current epoch number
        model (nn.Module): Model to evaluate
        val_loader (DataLoader): Validation loader
        num_classes (int): Number of classes in the dataset
        class_names (list): List of class names for reporting (optional, required if detailed_report is True)
        detailed_report (bool): Whether to provide detailed per-class results
        callbacks (list): List of callback functions to run during validation
        device (str): Device to run the evaluation on, default is 'cpu'
    
    Returns:
        float: Mean IoU for the validation set
        pd.DataFrame (optional): DataFrame with class names and their IoU if detailed_report is True
    '''
    if detailed_report and not class_names:
        raise ValueError("class_names must be provided if detailed_report is True")

    # Run validation begin callbacks
    for callback in callbacks:
        callback.on_validation_begin()

    model.eval()
    total_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

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
            total_confusion_matrix += utils.fast_hist(targets, predicted, num_classes)

            # Calculate True Positive on the confusion matrix
            TP = np.diag(total_confusion_matrix)
            # Calculate the Accuracy
            pixel_acc = np.sum(TP) / np.sum(total_confusion_matrix)
            # Loss
            loss = 1. - pixel_acc

            # Run validation batch end callbacks
            for callback in callbacks:
                callback.on_validation_batch_end(batch_idx, loss)

    # Calculate per-class IoU from the confusion matrix
    IoUs = utils.per_class_iou(total_confusion_matrix)
    mean_iou = np.nanmean(IoUs)  # Calculate mean IoU, ignoring NaNs
    print(f'Validation Mean IoU for Epoch {epoch + 1}: {mean_iou:.4f}')

    if detailed_report:
        # Create a DataFrame with class names and IoU values
        class_result_df = pd.DataFrame({'Class': class_names, 'IoU': [f'{iou:.4f}' for iou in IoUs]})
        print(class_result_df)

        # Run validation end callbacks with detailed data
        for callback in callbacks:
            callback.on_validation_end({
                'validation_mIoU': mean_iou
            }, data=class_result_df)

        return mean_iou, class_result_df
    else:
        # Run validation end callbacks without detailed data
        for callback in callbacks:
            callback.on_validation_end(mean_iou)

        return mean_iou


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


def adversarial_train(iterations : int ,epochs : int, generator : torch.nn.Module, discriminator : torch.nn.Module,
           generator_optimizer : torch.optim.Optimizer, discriminator_optimizer : torch.optim.Optimizer,
            source_dataloader : DataLoader, target_dataloader : DataLoader,
            generator_loss : torch.nn.Module, discriminator_loss : torch.nn.Module, lambda_ : float,
            gen_init_lr : float, gen_power : float, dis_power : float, dis_init_lr : float, lr_decay_iter : float, 
            num_classes : int, class_names : list[str], val_loader : DataLoader,do_validation : int = 1,
            device : str = 'cpu', when_print : int = 10, callbacks : list[Callback]  = []):
    

    # defining the target interpolation
    # target_interpolation = torch.nn.Upsample(size=(target_dataloader.dataset[0][1].shape[1],target_dataloader.dataset[0][1].shape[2]), mode='bilinear')
    
    for epoch in range(epochs):

        # setup callbacks
        for callback in callbacks:
            callback.on_train_begin()


        running_generator_source_loss = 0.0
        running_adversarial_loss = 0.0
        running_discriminator_source_loss = 0.0
        running_discriminator_target_loss = 0.0

        generator_correct = 0
        generator_total = 0

        generator.train()
        discriminator.train()

        dis_lr = utils.poly_lr_scheduler(discriminator_optimizer, dis_init_lr , epoch, lr_decay_iter, epochs, dis_power)
        # gen_lr = utils.poly_lr_scheduler(generator_optimizer, gen_lr , epoch, lr_decay_iter, epochs, gen_power)

        max_iter = epochs * iterations

        for i in tqdm(range(iterations),total=iterations,desc=f'Epoch {epoch}'):

            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            # ## lr_scheduler for the generator
            current_iter = epoch * iterations + i  # Calculate global iteration count across all epochs
            # Update learning rate
            if current_iter % lr_decay_iter == 0 and current_iter <= max_iter:
                gen_lr = utils.poly_lr_scheduler(generator_optimizer, gen_init_lr , current_iter, lr_decay_iter, max_iter, gen_power)

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

            # normalize the loss
            loss_gen_source = loss_gen_source / iterations
            # backward the generator loss        
            loss_gen_source.backward()
            running_generator_source_loss += loss_gen_source.item()

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

            # normalize the loss
            loss_adversarial = loss_adversarial / iterations

            running_adversarial_loss += loss_adversarial.item()

            # ! //////////////////////// -------------  Training the Discriminator  ------------- //////////////////////// !
            ## * The Discriminator weight now should be updated during the discriminator training !
            for param in discriminator.parameters():
                param.requires_grad = True

            # detaching the features to avoid the gradient flow to the generator
            source_features = source_features.detach()
            target_feature = target_feature.detach()
            ## * Train with the source data
            predicted_source_domain = discriminator(F.softmax(source_features,dim=1))
            source_mask = torch.ones(predicted_source_domain.size()).to(device)
            loss_disc_source = discriminator_loss(predicted_source_domain, source_mask)

            # normalize the loss
            loss_disc_source = loss_disc_source / iterations            

            loss_disc_source.backward()
            running_discriminator_source_loss += loss_disc_source.item()

            ## * Train with the target data
            predicted_target_domain = discriminator(F.softmax(target_feature,dim=1))
            target_mask = torch.zeros(predicted_target_domain.size()).to(device)
            loss_disc_target = discriminator_loss(predicted_target_domain, target_mask)

            # normalize the loss
            loss_disc_target = loss_disc_target / iterations            

            loss_disc_target.backward()
            running_discriminator_target_loss += loss_disc_target.item()

            # ! //////////////////////// -------------  Finalizations  ------------- //////////////////////// !

            # updating the generator weights
            generator_optimizer.step()
            discriminator_optimizer.step()

            generator_predictred = source_features.argmax(dim=1)    
            generator_correct += generator_predictred.eq(source_label).sum().item()

            generator_total += source_label.size(0) * source_label.size(1) * source_label.size(2)  # Total number of pixels

            ## * ---------------------- Loggings ---------------------- * ##
            for callback in callbacks:
                callback.on_batch_end(i, {
                    'loss_gen_source': loss_gen_source.item(),
                    'loss_adversarial': loss_adversarial.item(),
                    'loss_disc_source': loss_disc_source.item(),
                    'loss_disc_target': loss_disc_target.item(),
                })


        print(f'Epoch Results {epoch}')
        utils.tabular_print({
            'loss_gen_source': running_generator_source_loss/iterations,
            'loss_adversarial': running_adversarial_loss/iterations,
            'loss_disc_source': running_discriminator_source_loss/iterations,
            'loss_disc_target': running_discriminator_target_loss/iterations,
            'Genrator Accuracy': (100. * generator_correct / generator_total),
            'dis_lr': dis_lr if dis_lr else -1,
            'gen_lr': gen_lr if gen_lr else -1,
        })  

        for callback in callbacks:
            callback.on_epoch_end(epoch, {
                'dis_lr': dis_lr if dis_lr else -1,
                'gen_lr': gen_lr if gen_lr else -1,
                'Genrator Accuracy': 100. * generator_correct / generator_total,
            })

        if epoch % do_validation == 0 and do_validation != 0:
            print('-'*50, 'Validation', '-'*50)
            val_GTA5(epoch, generator, val_loader, num_classes, class_names, callbacks, device=device)
            print('-'*100)


    for callable in callbacks:
        callable.on_train_end()
    

# second implementation of the adversarial training
def adversarial_train_2(iterations : int ,epochs : int, generator : torch.nn.Module, discriminator : torch.nn.Module,
           generator_optimizer : torch.optim.Optimizer, discriminator_optimizer : torch.optim.Optimizer,
            source_dataloader : DataLoader, target_dataloader : DataLoader,
            generator_loss : torch.nn.Module, discriminator_loss : torch.nn.Module, lambda_ : float,
            gen_init_lr : float, gen_power : float, dis_power : float, dis_init_lr : float, lr_decay_iter : float, 
            num_classes : int, class_names : list[str], val_loader : DataLoader,
            do_validation : int = 1,device : str = 'cpu', when_print : int = 10, callbacks : list[Callback]  = []):
    
    """
    Train a generator and discriminator in an adversarial setting for domain adaptation.

    This function implements the adversarial training loop for a GAN-like model, where the generator learns to generate
    realistic samples that can fool the discriminator into classifying source domain samples as target domain samples.

    Args:
        iterations (int): Number of iterations per epoch.
        epochs (int): Total number of training epochs.
        generator (torch.nn.Module): The generator network to be trained.
        discriminator (torch.nn.Module): The discriminator network to be trained.
        generator_optimizer (torch.optim.Optimizer): Optimizer for the generator.
        discriminator_optimizer (torch.optim.Optimizer): Optimizer for the discriminator.
        source_dataloader (DataLoader): DataLoader for the source domain.
        target_dataloader (DataLoader): DataLoader for the target domain.
        generator_loss (torch.nn.Module): Loss function for the generator.
        discriminator_loss (torch.nn.Module): Loss function for the discriminator.
        lambda_ (float): Weight for the adversarial loss component in the generator's total loss.
        gen_init_lr (float): Initial learning rate for the generator.
        gen_power (float): Power factor for the polynomial learning rate decay for the generator.
        dis_power (float): Power factor for the polynomial learning rate decay for the discriminator.
        dis_init_lr (float): Initial learning rate for the discriminator.
        lr_decay_iter (float): Interval in iterations for learning rate decay.
        num_classes (int): Number of output classes for the segmentation task.
        class_names (list[str]): List of class names corresponding to the output classes.
        val_loader (DataLoader): DataLoader for the validation set.
        do_validation (int, optional): Frequency (in epochs) to perform validation. Default is 1.
        device (str, optional): Device to run the training on ('cpu' or 'cuda'). Default is 'cpu'.
        when_print (int, optional): Frequency (in iterations) for printing the progress. Default is 10.
        callbacks (list[Callback], optional): List of callback objects to handle events during training. Default is an empty list.

    Returns:
        None

    Notes:
        - The source domain typically represents synthetic or less complex data, while the target domain represents real-world data.
        - Real and fake labels for the discriminator are created for the adversarial training process.
        - This implementation uses an `adaptive average pooling` to ensure the size compatibility between different domains.
        - The learning rate scheduler uses a `polynomial decay` strategy based on the current iteration.
        - Generator and discriminator are updated alternately within each iteration of training.
        - Validation and callback mechanisms are incorporated to monitor training progress.
    """

    ## Notice: 
    # real_seg ---> Target Domain
    # fake_seg ---> Source Domain

    # defining the target interpolation
    # target_interpolation = torch.nn.Upsample(size=(target_dataloader.dataset[0][1].shape[1],target_dataloader.dataset[0][1].shape[2]), mode='bilinear')
    
    for epoch in range(epochs):

        generator.train()
        discriminator.train()

        running_generator_source_loss = 0.0
        running_adversarial_loss = 0.0
        running_discriminator_source_loss = 0.0
        running_discriminator_target_loss = 0.0
        running_d_total_loss = 0.0
        running_g_total_loss = 0.0

        generator_correct = 0
        generator_total = 0

        
        # gen_lr = utils.poly_lr_scheduler(generator_optimizer, gen_lr , epoch, lr_decay_iter, epochs, gen_power)

        max_iter = epochs * iterations

        for i in tqdm(range(iterations),total=iterations,desc=f'Epoch {epoch}'):

            # ! //////////////////////// -------------  Initialization  ------------- //////////////////////// !
            # defining source and target data
            source_image, source_label = next(iter(source_dataloader))
            target_image, _ = next(iter(target_dataloader))

            source_image, source_label = source_image.to(device), source_label.to(device)
            source_label = source_label.squeeze(1) # removing the channel dimension
            target_image = target_image.to(device)

            target_size = target_image.size()

            # !  !! Warning: The Batch size of the source and target should be the same !!
            real_labels = torch.ones(target_size[0],1,1,1).to(device).requires_grad_(False)
            fake_labels = torch.zeros(target_size[0],1,1,1).to(device).requires_grad_(False)
            # ----------------------
            # ! Train the Generator
            # ----------------------
            # zero the gradients
            generator_optimizer.zero_grad()
            # ## lr_scheduler for the generator
            current_iter = epoch * iterations + i  # Calculate global iteration count across all epochs
            # Update learning rate
            if current_iter % lr_decay_iter == 0 and current_iter <= max_iter:
                    dis_lr = utils.poly_lr_scheduler(discriminator_optimizer, dis_init_lr , current_iter, lr_decay_iter, max_iter, dis_power)
                    gen_lr = utils.poly_lr_scheduler(generator_optimizer, gen_init_lr , current_iter, lr_decay_iter, max_iter, dis_power)
            # train on the source data
            fake_seg = generator(source_image)

            if isinstance(fake_seg,tuple):
                g_loss_seg = generator_loss(fake_seg[0], source_label)
                g_loss_seg += generator_loss(fake_seg[1], source_label)
                g_loss_seg += generator_loss(fake_seg[2], source_label)
                fake_seg = fake_seg[0]
            else:
                g_loss_seg = generator_loss(fake_seg, source_label)

            # for loggging the accuracy
            generator_predictred = fake_seg.argmax(dim=1)
            generator_correct += generator_predictred.eq(source_label).sum().item()
            generator_total += source_label.size(0) * source_label.size(1) * source_label.size(2)  # Total number of pixels

            # ! Adversarial loss
            real_seg = generator(target_image)
            if isinstance(real_seg,tuple):
                real_seg = real_seg[0]

            real_seg = F.adaptive_avg_pool2d(real_seg, (target_size[2], target_size[3]))
            d_real_output = discriminator(F.softmax(real_seg,dim=1))
            loss_adv = discriminator_loss(d_real_output, fake_labels)
            # Total loss for the generator
            
            # lambda scheduling 
            lambda_adv = max(lambda_, (lambda_*10) - 0.001 * epoch)  # Decrease over time
            g_loss = g_loss_seg + lambda_adv * loss_adv
            # backward the loss
            g_loss.backward()
            # update the generator weights
            generator_optimizer.step()
            # ! //////////////////////// -------------  Logging  ------------- //////////////////////// !
            running_generator_source_loss += g_loss_seg.item()
            running_adversarial_loss += loss_adv.item()
            running_g_total_loss += g_loss.item()
            # ---------------------- 
            # ! Train the Discriminator
            # ----------------------
            # zero the gradients
            discriminator_optimizer.zero_grad()
            # disable the gradient flow to the generator
            with torch.no_grad():
                # the fake segmentation from the source image
                fake_seg = generator(source_image)
                if isinstance(fake_seg,tuple):
                    fake_seg = fake_seg[0]

                fake_seg = F.adaptive_avg_pool2d(fake_seg, (target_size[2], target_size[3]))
                # forward pass the fake segmentation to the discriminator
                # Real segmentation from the target image (since the target image is from the Cityscapes Real-Wold dataset)
                real_seg = generator(target_image)
                if isinstance(real_seg,tuple):
                    real_seg = real_seg[0]

                real_seg = F.adaptive_avg_pool2d(real_seg, (target_size[2], target_size[3]))

            # forward pass the real and fake segmentation to the discriminator
            # * - [1] This is an idea that we may able to train the discriminator with the real_seg from source label
            # *       Not with the output of the generator.
            d_real_output = discriminator(F.softmax(real_seg.detach(),dim=1)).requires_grad_(True)
            d_fake_output = discriminator(F.softmax(fake_seg.detach(),dim=1)).requires_grad_(True)

            # calculate the loss for the discriminator
            d_real_loss = discriminator_loss(d_real_output, real_labels)
            d_fake_loss = discriminator_loss(d_fake_output, fake_labels)

            # calculate the total loss for the discriminator (Average the loss)
            # * - [3] You might want to use the average loss instead of the sum of the loss
            d_loss = (d_real_loss + d_fake_loss) 
            # backward the loss
            d_loss.backward()
            # update the discriminator weights
            discriminator_optimizer.step()
            # ! //////////////////////// -------------  Logging  ------------- //////////////////////// !
            running_discriminator_target_loss += d_real_loss.item()
            running_discriminator_source_loss += d_fake_loss.item()
            running_d_total_loss += d_loss.item()
             
        print(f'Epoch Results {epoch}')
        utils.tabular_print({
            'Genrator Accuracy': (100. * generator_correct / generator_total),
            'dis_lr': dis_lr if dis_lr else -1,
            'gen_lr': gen_lr if gen_lr else -1,
        })

        for callback in callbacks:
            callback.on_epoch_end(epoch, {
                'dis_lr': dis_lr if dis_lr else -1,
                'gen_lr': gen_lr if gen_lr else -1,
                'loss_gen_source': running_generator_source_loss/iterations,
                'loss_adversarial': running_adversarial_loss/iterations,
                'loss_disc_source': running_discriminator_source_loss/iterations,
                'loss_disc_target': running_discriminator_target_loss/iterations,
                'loss_disc_total': running_d_total_loss/iterations,
                'loss_gen_total': running_g_total_loss/iterations,
                'Genrator Accuracy': 100. * generator_correct / generator_total,
            })
            

        if do_validation != -1 and epoch % do_validation == 0 and epoch != 0:
            print('-'*50, 'Validation', '-'*50)
            val_GTA5(epoch, generator, val_loader, num_classes, class_names, callbacks, device=device)
            print('-'*100)
        
    for callable in callbacks:
        callable.on_train_end()



