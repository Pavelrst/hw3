from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .autoencoder import EncoderCNN, DecoderCNN


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        # To extract image features you can use the EncoderCNN from the VAE
        # section or implement something new.
        # You can then use either an affine layer or another conv layer to
        # flatten the features.
        # ====== YOUR CODE: ======
        ndf = in_size[1]  # nit a must
        # input is (nc) x 64 x 64
        modules = []

        #class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        #stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

        # We are used this manual:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        modules.append(nn.Conv2d(in_size[0], ndf, 4, 2, 1, bias=False))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        modules.append(nn.BatchNorm2d(ndf * 2))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        modules.append(nn.BatchNorm2d(ndf * 4))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        modules.append(nn.BatchNorm2d(ndf * 8))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))

        self.discriminator = nn.Sequential(*modules)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (aka logits, not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        # No need to apply sigmoid to obtain probability - we'll combine it
        # with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        output = self.discriminator(x)
        y=output.view(-1, 1)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        # To combine image features you can use the DecoderCNN from the VAE
        # section or implement something new.
        # You can assume a fixed image size.
        # ====== YOUR CODE: ======





        # normalize the images between -1 and 1
        # Tanh as the last layer of the generator output
        # Use leaky ReLU

        #self.generator = nn.Sequential(
        #    #z_dim: Dimension of latent space.
        #    nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
        #    nn.BatchNorm2d(ngf * 8),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #    nn.BatchNorm2d(ngf * 4),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #    nn.BatchNorm2d(ngf * 2),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        #    nn.BatchNorm2d(ngf),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1, bias=False),
        #    nn.Tanh()
        #)

        #class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        #stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)


        # We are used this pytorch manual:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        ngf = 64
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        #class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        #stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should have
        gradients or not.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        # Generate n latent space samples and return their reconstructions.
        # Don't use a loop.
        # ====== YOUR CODE: ======

        # sample noise to generate fake images.
        z = torch.randn(n, self.z_dim, device=device)

        # generate fake
        if with_grad == True:
            samples = self.forward(z)
        else:
            samples = self.forward(z).data
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        # Don't forget to make sure the output instances have the same scale
        # as the original (real) images.
        # ====== YOUR CODE: ======
        z_unfl = z.view(-1, self.z_dim, 1, 1)
        x = self.generator(z_unfl)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO: Implement the discriminator loss.
    # See torch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======

    # y_data: output class-scores of discriminator for real data.
    # y_generated: output class-scores of discriminator for fake data.
    # data_label: ground truth class of same real data.

    # Fuzzy label for real data
    y_data_fuzzy = torch.rand(y_data.shape[0]) * label_noise + data_label - label_noise / 2

    # Fuzzy label for fake data - always near zero
    data_label = 0
    y_generated_fuzzy = torch.rand(y_generated.shape[0]) * label_noise + data_label - label_noise / 2

    discriminator_loss = nn.BCEWithLogitsLoss()
    loss_data = discriminator_loss(y_data, y_data_fuzzy)
    loss_generated = discriminator_loss(y_generated, y_generated_fuzzy)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    # TODO: Implement the Generator loss.
    # Think about what you need to compare the input to, in order to
    # formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    #print ("y_generated=",y_generated)
    label = torch.FloatTensor(y_generated.shape[0])
    label.data.resize_(y_generated.shape[0]).fill_(data_label)
    #print ("label=",label)

    # y_generated is what we get from discriminator. It tells if it real or fake.
    generator_loss = nn.BCEWithLogitsLoss()
    loss = generator_loss(y_generated, label)
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======

    #optimizer_D.zero_grad()
    # Measure discriminator's ability to classify real from generated samples
    #real_loss = adversarial_loss(discriminator(real_imgs), valid)
    #fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
    #d_loss = (real_loss + fake_loss) / 2
    #d_loss.backward()
    #optimizer_D.step()

    dsc_optimizer.zero_grad()

    real_labels = dsc_model.forward(x_data).view(-1)

    fake_imgs = gen_model.sample(x_data.shape[0], with_grad=False)
    fake_labels = dsc_model.forward(fake_imgs).view(-1)

    dsc_loss = dsc_loss_fn(real_labels, fake_labels)
    dsc_loss.backward()

    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======

    #optimizer_G.zero_grad()
    ## Sample noise as generator input
    #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    ## Generate a batch of images
    #gen_imgs = generator(z)
    ## Loss measures generator's ability to fool the discriminator
    #g_loss = adversarial_loss(discriminator(gen_imgs), valid)
    #g_loss.backward()
    #optimizer_G.step()

    gen_optimizer.zero_grad()

    # Generate a batch of images
    fake_imgs = gen_model.sample(x_data.shape[0], with_grad=True)
    fake_labels = dsc_model.forward(fake_imgs).view(-1)

    # Loss measures generator's ability to fool the discriminator
    gen_loss = gen_loss_fn(fake_labels)
    gen_loss.backward()

    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()

