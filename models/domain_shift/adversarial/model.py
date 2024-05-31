# from functions import DomainDiscriminator
# import sys
# sys.path.append('../../../')

# class AdversarialTrian:
#     def __init__(self, num_classes=19, with_grl = False,lambda_ : float = 0.1) -> None:
#         self.num_classes = num_classes
#         self.with_grl = with_grl
#         self.lambda_ = lambda_
#         self.domain_discriminator = DomainDiscriminator(num_classes=self.num_classes, with_grl=self.with_grl, lambda_=self.lambda_)
#         self.up_sampler = UpSampler(num_classes=self.num_classes)
#         self.domain_discriminator = self.domain_discriminator.to(device)
#         self.up_sampler = self.up_sampler.to(device)
#         self.domain_discriminator_optimizer = torch.optim.Adam(self.domain_discriminator.parameters(), lr=1e-4)
#         self.up_sampler_optimizer = torch.optim.Adam(self.up_sampler.parameters(), lr=1e-4)
#         self.domain_discriminator_criterion = nn.BCEWithLogitsLoss()

# def train():
#     pass