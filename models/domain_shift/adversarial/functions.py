import torch
from torch import nn
import warnings
import sys
sys.path.append('../../../')
from models.bisenet.build_bisenet import BiSeNet
from torch.utils.data import DataLoader
import torch.nn.functional as F

warnings.filterwarnings(action='ignore')
    
from torch.autograd import Function
# Define the Gradient Reversal Layer
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
    
class UpSampler(nn.Module):

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=8, mode='bilinear') # torch.Size([4, 19, 720, 1280])
        x = self.conv(x)
        return x

class DomainDiscriminator(nn.Module):
    '''
    The network consists of 5 convolution layers with kernel 4 x 4 and stride of 2,
    where the channel number is {64, 128, 256, 512, 1}, respectively.
    with the input size [4, 19, 720, 1280] (batch size, channel, height, width).
    Except for the last layer, each convolution layeris followed by a leaky ReLU [27] parameterized by 0.2.
    and for the last layer, a softmax function is applied to output the probability of the input image from the source domain or the target domain.
    No batch normalization is used in the discriminator.
    '''
    def __init__(self, num_classes=19, with_grl = False,lambda_ : float = 0.1) -> None:
        super(DomainDiscriminator, self).__init__()

        self.with_grl = with_grl
        self.lambda_ = lambda_

        self.conv1 = nn.Conv2d(19, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        # defining the upsampler to interpolate the output to the same size as the input
        

        # interpolate the output to the same size as the input
        
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.classifier(x)
        
        if self.with_grl:
            x = GradientReversalFunction.apply(x, self.lambda_)
        
        return x
    

class DANN(nn.Module):
    def __init__(self, num_classes=19, alpha=1.0):
        super(DANN, self).__init__()
        self.alpha = alpha
        # defining the Generator
        self.segmentation_features = BiSeNet(num_classes=num_classes,context_path='resnet18',with_interpolation=False)
        # define the segmentation head (classifier)
        self.segments = UpSampler(num_classes=num_classes)
        # defining the Discriminator
        self.discriminator = DomainDiscriminator()


    def forward(self, x):
        # input x is the input image
        # output of the generator and the discriminator
        features = self.segmentation_features(x)
        segment = self.segments(features)
        reverse_feature = GradientReversalFunction.apply(features, self.alpha)
        dis = self.discriminator(reverse_feature)
        return segment, dis
    

def train(iterations : int ,epoch : int, generator : torch.nn.Module, discriminator : torch.nn.Module,
           generator_optimizer : torch.optim.Optimizer, discriminator_optimizer : torch.optim.Optimizer,
            source_dataloader : DataLoader, target_dataloader : DataLoader,
            generator_loss : torch.nn.Module, discriminator_loss : torch.nn.Module, 
            discriminator_interpolator : torch.nn.Module, image_inter_size : tuple,
            device : str = 'cpu', when_print : int = 10):
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
    when_print : int
        On which iteration to print the loss values (default is 10). it should be less than the iterations
    
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
        loss_adv_gen = discriminator_loss(disc_target_preds_gen, torch.ones(batch_size_target, 1, *image_inter_size).to(device))
        
        # Total generator loss
        loss_gen = gen_source_loss + loss_adv_gen
        loss_gen.backward()
        generator_optimizer.step()
        
        running_loss_gen += loss_gen.item()

        if i % when_print == 0 and i != 0:
            print(f'Iteration {i}', f"Generator Loss: {running_loss_gen/iterations:.4f}, "
              f"Discriminator Loss: {running_loss_disc/iterations:.4f}")
