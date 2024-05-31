import torch
from torch.autograd import Function
###### ------------------ Gradient Reversal Layer ------------------ ######
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
    return GradientReversalFunction.apply(x, alpha)

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
    return GradientReversalFunction.apply(x, alpha)

###### ------------------ Adversarial Training ------------------ ######

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

###### ------------------ Segmentation ------------------ ######

# Define a simple segmentation network
class SimpleSegNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSegNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#####------------------ Discriminator ------------------ ######

# Define a simple discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


#####------------------ Adversarial Training ------------------ ######

# Adversarial training function
def train_adversarial_with_grl(seg_net, disc_net, source_loader, target_loader, num_classes, num_epochs=10, alpha=1.0):
    seg_optimizer = optim.Adam(seg_net.parameters(), lr=0.001)
    disc_optimizer = optim.Adam(disc_net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

    for epoch in range(num_epochs):
        seg_net.train()
        disc_net.train()

        for (source_data, source_label), (target_data, _) in zip(source_loader, target_loader):
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()

            # Train segmentation network on source data
            seg_optimizer.zero_grad()
            source_pred = seg_net(source_data)
            seg_loss = criterion(source_pred, source_label)
            seg_loss.backward()
            seg_optimizer.step()

            # Train discriminator network
            disc_optimizer.zero_grad()

            source_features = seg_net.features(source_data).detach()
            target_features = seg_net.features(target_data).detach()

            source_disc_pred = disc_net(source_features)
            target_disc_pred = disc_net(target_features)

            source_disc_label = torch.ones_like(source_disc_pred)
            target_disc_label = torch.zeros_like(target_disc_pred)

            disc_loss = bce_loss(source_disc_pred, source_disc_label) + bce_loss(target_disc_pred, target_disc_label)
            disc_loss.backward()
            disc_optimizer.step()

            # Train segmentation network with GRL to fool discriminator
            seg_optimizer.zero_grad()
            target_features = grad_reverse(seg_net.features(target_data), alpha)
            target_disc_pred = disc_net(target_features)
            adv_loss = bce_loss(target_disc_pred, target_disc_label)
            adv_loss.backward()
            seg_optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Segmentation Loss: {seg_loss.item():.4f}, '
                  f'Discriminator Loss: {disc_loss.item():.4f}, Adversarial Loss: {adv_loss.item():.4f}')

# Example usage
if __name__ == '__main__':
    num_classes = 21  # Number of segmentation classes
    batch_size = 8
    num_epochs = 10

    # Data loaders (replace with actual data loaders)
    source_loader = DataLoader(datasets.FakeData(transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(datasets.FakeData(transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

    seg_net = SimpleSegNet(num_classes).cuda()
    disc_net = Discriminator().cuda()

    train_adversarial_with_grl(seg_net, disc_net, source_loader, target_loader, num_classes, num_epochs)
