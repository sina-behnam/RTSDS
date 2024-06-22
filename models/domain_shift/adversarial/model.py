import torch
from torch import nn
import warnings

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
        # For simplicity, we add a adaptive average pooling layer to make the output size 1 x 1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.classifier(x)
        x = self.avgpool(x)
        if self.with_grl:
            x = GradientReversalFunction.apply(x, self.lambda_)
        
        return x
