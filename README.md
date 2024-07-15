# Semantic Segmentation and Domain Adaptation on Cityscapes and GTA5 datasets



#### Overview

This repository contains the code for semantic segmentation and domain adaptation on Cityscapes and GTA5 datasets. The code is implemented in PyTorch and uses the DeepLabV and BieSNet model for semantic segmentation. and provided two different datasets as real and synthetic data. The `main.py` script serves as the primary executable to start the training process, guided by configurations specified in the `config.yaml` file.

#### Prerequisites
1. **Dependencies**: Install the required Python packages using the following command:
   ```sh
   pip install -r requirements.txt
   ```

#### Configuration

The `config.yaml` file contains all the necessary configurations for data, model, training, augmentation, and callbacks. Below is a detailed description of the configuration parameters:

##### Data Configuration

- **Cityscapes Dataset**:
  - `images_train_dir`: Directory for training images.
  - `images_val_dir`: Directory for validation images.
  - `segmentation_train_dir`: Directory for training segmentation labels.
  - `segmentation_val_dir`: Directory for validation segmentation labels.
  - `image_size`: Tuple specifying the image size.
  - `num_classes`: Number of classes.
  - `batch_size`: Batch size for training.
  - `num_workers`: Number of worker threads for data loading.

- **GTA5 Modified Dataset**:
  - Similar parameters as above, but specific to the GTA5 dataset.

##### Meta Data

- `class_names`: List of class names used for segmentation.

##### Model Configuration

- **DeepLab**:
  - `backbone`: Backbone network architecture.
  - `output_stride`: Output stride for the model.
  - `num_classes`: Number of classes.
  - `pretrained`: Boolean indicating if a pretrained model should be used.
  - `pretrained_path`: Path to the pretrained model file.
  - `optimizer`: Optimizer settings.
  - `criterion`: Loss function settings.

- **BiSeNet**:
  - Similar parameters as for DeepLab.

- **Adversarial Model**:
  - Configuration for the generator and discriminator used in domain adaptation.

##### Training Configuration

- **Segmentation**:
  - Training settings specific to segmentation tasks.

- **Domain Adaptation**:
  - Training settings specific to domain adaptation tasks.

##### Augmentation Configuration

- Various data augmentation settings such as Gaussian blur and horizontal flip.

##### Callbacks Configuration

- **Model Checkpoint**:
  - Settings for saving model checkpoints.
  
- **Early Stopping**:
  - Settings for early stopping based on validation loss.
  
- **Logging**:
  - Settings for logging training progress with tools like Weights & Biases.
  
- **Images Plots**:
  - Settings for saving image plots during training.

##### Device

- `device`: Specifies whether to use `cpu` or `cuda` (GPU) for training.

#### Usage

To start the training process, run the following command:

```sh
python main.py --config config.yaml
```

#### Arguments

For getting help about the arguments, run the following command:

```sh
python main.py --help
```

#### Main Script (`main.py`)

The main script is responsible for:

1. **Loading Configuration**: Reads the configuration from `config.yaml`.
2. **Initializing Model**: Sets up the model architecture based on the configuration.
3. **Data Loading**: Prepares the data loaders for training and validation datasets.
4. **Training Loop**: Executes the training loop, including forward and backward passes, loss calculation, and optimizer updates.
5. **Validation**: Performs validation at specified intervals and logs the results.
6. **Callbacks**: Handles callbacks such as model checkpointing, early stopping, and logging.

---

## Model initialization

In the following link you can find the pretrained weights for DeepLab.

**DeepLab petrained weights**: https://drive.google.com/file/d/1ZX0UCXvJwqd2uBGCX7LI2n-DfMg3t74v/view?usp=sharing


## Datasets

To download the dataset use the following download links.

**Cityscapes**: https://drive.google.com/file/d/1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2/view?usp=sharing

**GTA5**: https://drive.google.com/file/d/1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23/view?usp=sharing

## GTA5: label color mapping

Plese refer to this link to convert GTA5 labels in the same format of Cityscapes: https://github.com/sarrrrry/PyTorchDL_GTA5/blob/master/pytorchdl_gta5/labels.py

## FLOPs and parameters

First install fvcore with this command:
```bash
!pip install -U fvcore
```

To calculate the FLOPs and number of parameters please use this code:
```python
from fvcore.nn import FlopCountAnalysis, flop_count_table

# -----------------------------
# Initizialize your model here
# -----------------------------

height = ...
width = ...
image = torch.zeros((3, height, width))

flops = FlopCountAnalysis(model, image)
print(flop_count_table(flops))
```
Reference: https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md

## Latency and FPS

Please refer to this pseudo-code for latency and FPS calculation.

> $\texttt{image} \gets \texttt{random(3, height, width)}$\
$\texttt{iterations} \gets 1000$\
$\texttt{latency} \gets \texttt{[]}$\
$\texttt{FPS} \gets \texttt{[]}$ \
repeat $\texttt{iterations}$ times \
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{start = time.time()}$\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{output = model(image)}$\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{end = time.time()}$\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{latency}_i \texttt{ = end - start} $\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{latency.append(latency}_i \texttt{}) $\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{FPS}_i = \frac{\texttt{1}}{\texttt{latency}_i}$\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\texttt{FPS.append(FPS}_i \texttt{})$    
end      
> $\texttt{meanLatency}  \gets \texttt{mean(latency)*1000}$\
$\texttt{stdLatency} \gets \texttt{std(latency)*1000}$\
$\texttt{meanFPS} \gets \texttt{mean(FPS)}$\
$\texttt{stdFPS} \gets \texttt{std(FPS)}$
