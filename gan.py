import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from utils import Logger


def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((.5,), (.5,))]
    )
    out_dir = '.dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)


# Load data
data = mnist_data()

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

# Num batches
num_batches = len(data_loader)


def images_to_vector(images):
    return images.view(images.size(0), 784)


def vector_to_images(vector):
    return vector.view(vector.size(0), 1, 28, 28)


def noise(size):
    """Generate a size-d vector of gaussian sampled random values"""

    n = Variable(torch.randn(size, 100))
    return n


# Networks
class DiscriminatorNet(nn.Module):
    """A three hidden-layer discriminative network
    """
    def __init__(self):
        super().__init__()
        n_features = 784
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class GeneratorNet(nn.Module):
    def __init__(self):
        super().__init__()
        n_features = 100
        n_out = 784

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


discriminator = DiscriminatorNet()
generator = GeneratorNet()

# Optimization
d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4)
g_optimizer = optim.Adam(generator.parameters(), lr=2e-4)
loss = nn.BCELoss()


def ones_target(size):
    """Tensor containing ones, with shape = size"""

    data = Variable(torch.ones(size, 1))
    return data


def zeros_target(size):
    """Tensor containing zeros, with shape = size"""

    data = Variable(torch.zeros(size, 1))
    return data


def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    N = fake_data.size(0)

    # Reset gradients
    optimizer.zero_grad()

    # Sample noise and generate fake data
    prediction = discriminator(fake_data)

    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()

    # Update weights with gradients
    optimizer.step()

    # Return error
    return error


# Testing
num_test_samples = 16
test_noise = noise(num_test_samples)

# Training
# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST')

# Total number of epoches to train
num_epoches = 200

for epoch in range(num_epoches):
    for n_batch, (real_batch, _) in enumerate(data_loader):
        N = real_batch.size(0)

        # 1. Train Discriminator
        real_data = images_to_vector(real_batch)

        # Generate fake data and detach
        # (so gradient are not calculated for generator)
        fake_data = generator(noise(N)).detach()

        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N))

        # Train G
        g_error = train_generator(g_optimizer, fake_data)

        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # Display progress every few batches
        if n_batch % 100 == 0:
            test_images = vector_to_images(generator(test_noise))
            test_images = test_images.data

            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches)

            # Display status log
            logger.display_status(epoch, num_epoches, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake)
