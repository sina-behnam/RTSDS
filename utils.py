import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

# Cityscapes color palette for 19 classes + unlabeled
cityscapes_color_palette = {
    0: [128, 64, 128],   # road
    1: [244, 35, 232],   # sidewalk
    2: [70, 70, 70],     # building
    3: [102, 102, 156],  # wall
    4: [190, 153, 153],  # fence
    5: [153, 153, 153],  # Pole
    6: [250, 170, 30],   # traffic light
    7: [220, 220,  0],   # traffic sign
    8: [107, 142, 35],   # vegetation
    9: [152, 251, 152],  # terrain
    10: [70, 130, 180],  # sky
    11: [220, 20, 60],   # person
    12: [255, 0, 0],     # rider
    13: [0,  0, 142],    # car
    14: [0,  0, 70],     # truck
    15: [0, 60, 100],    # bus
    16: [0, 80, 100],    # train
    17: [0, 0, 230],     # motorcycle
    18: [119, 11, 32],   # bicycle
}

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr
    # return lr


def fast_hist(a, b, n):
    '''
    a and b are label and prediction respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


# To perform transformation in the dataset
class IntRangeTransformer:
    def __init__(self, min_val=0, max_val=255):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample):
        # Perform the transformation
        sample = torch.clamp(sample, self.min_val, self.max_val)  # Clamp values to the range
        return sample.long()  # Cast to torch.long
    
def tabular_print(log_dict):
    df = pd.DataFrame({**log_dict}, index=[0])

    # check if the PrettyTable module is available
    try:
        from prettytable import PrettyTable
    except ImportError:
        # log a warning once if the module is not available
        if not hasattr(tabular_print, 'warned'):
            print('PrettyTable is not available. Falling back to printing the dataframe.', file=sys.stderr)
            tabular_print.warned = True
        print(df)
        return
    
    x = PrettyTable()
    for col in df.columns:
        x.add_column(col, df[col].values)
    print(x)


def forModel(model,device):
    if device == 'cuda':
        model = model.cuda()  # Move model to GPU if available
        print('The number of cuda GPUs : ',torch.cuda.device_count())

    # If you have multiple GPUs available
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model) # Use DataParallel to wrap your model
        
    return model

def latency(model,device = 'cpu'):
    # Latency and FPS Calculation
    iterations = 1000
    latency = []
    FPS = []

    for _ in range(iterations):
        image = torch.randn((4, 3, 512, 1024)).to(device)
        start = time.time()
        with torch.no_grad():  # Ensure no gradients are computed for speed
            output = model(image)
        end = time.time()

        latency_i = end - start
        latency.append(latency_i)
        FPS_i = 1 / latency_i
        FPS.append(FPS_i)

    # Calculate mean and standard deviation for latency and FPS
    mean_latency =torch.mean(latency) * 1000  # Convert to milliseconds
    std_latency =torch.stdev(latency) * 1000
    mean_FPS =torch.mean(FPS)
    std_FPS =torch.stdev(FPS)

    print(f"Mean Latency: {mean_latency:.2f} ms, Std Latency: {std_latency:.2f} ms")
    print(f"Mean FPS: {mean_FPS:.2f}, Std FPS: {std_FPS:.2f}")


# FLOP Calculation
def flop(model,device = 'cpu'):
    image = torch.zeros((4, 3, 512, 1024)).to(device) # Check if we should change something
    flops = FlopCountAnalysis(model, image)
    print(flop_count_table(flops))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def apply_cityscapes_color_map(segmentation_map, color_palette):
    """Apply the Cityscapes color map to a segmentation map."""
    h, w = segmentation_map.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for key, color in color_palette.items():
        color_image[segmentation_map == key] = color
    return color_image


def rescale_for_display(input_tensor):
    """ Rescale the input tensor for display between 0 and 1 """
    min_val = input_tensor.min()
    max_val = input_tensor.max()
    rescaled_tensor = (input_tensor - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    return rescaled_tensor


def visualize_first_image_from_batches(inputs_list, targets_list, predictions, num_batches=5):
    """ Visualizes the first image from each batch for inputs, targets, and predictions. """
    num_batches = min(num_batches, len(inputs_list))
    fig, axes = plt.subplots(nrows=num_batches, ncols=3, figsize=(18, num_batches * 6))
    
    for idx in range(num_batches):
        ax = axes[idx] if num_batches > 1 else axes
        
        # Extract the first image from each batch
        input_tensor = inputs_list[idx][0]
        
        # Normalize for visualization
        input_tensor = rescale_for_display(input_tensor)
        
        # Enhance image contrast
        input_img = to_pil_image(input_tensor)
        
        ax[0].imshow(input_img)
        ax[0].set_title('Input Image')
        ax[0].axis('off')

        # Assuming targets and predictions are already in H, W or 1, H, W format
        target_img = targets_list[idx][0].squeeze(0).numpy()
        prediction_img = predictions[idx][0].squeeze(0).numpy()

        # Applying color maps
        colored_target = apply_cityscapes_color_map(target_img, cityscapes_color_palette)
        colored_prediction = apply_cityscapes_color_map(prediction_img, cityscapes_color_palette)
        
        ax[1].imshow(colored_target)
        ax[1].set_title('Ground Truth')
        ax[1].axis('off')

        ax[2].imshow(colored_prediction)
        ax[2].set_title('Prediction')
        ax[2].axis('off')

    plt.tight_layout()
    plt.show()