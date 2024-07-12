import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import time


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