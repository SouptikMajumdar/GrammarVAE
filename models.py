import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import seaborn as sns
import torchvision.datasets as datasets
from torchvision import transforms
sns.set()
torch.manual_seed(42) # Setting the seed
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.dropout1 = nn.Dropout(p=0.25)
        self.pool1 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64,64,3)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, 256)
        self.out = nn.Linear(256,num_outputs)
        self.dropout2 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        #Added for easier forward pass:
        self.encoder = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256,num_outputs),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.relu(self.conv1(x)) 
        x = self.relu(self.conv2(x)) 
        x = self.dropout1(x)
        x = self.pool1(x)
        x = self.relu(self.conv3(x)) 
        x = self.relu(self.conv4(x)) 
        x = self.dropout1(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.dropout2(x)
        x = self.out(x)
        #print(x.shape)
        return x


#Creation of Auto Encoder Class:
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        #Encoder:
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(784, 32)
        )

        #Decoder:
        self.decoder = nn.Sequential(
            nn.Linear(32, 784),
            nn.Unflatten(dim=1,unflattened_size=(16,7,7)),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #print(x.shape)
        Z = self.encoder(x)
        x = self.decoder(Z)
        #print(x.shape)
        return x


import torch.nn.functional as F
class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        #Convolutional Layer: Size Calculation:
        ''' HL = Floor(HL-1 +2P - (KL-1).D-1 / S) + 1 '''


        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3), # 28x28x1 -> 26x26x16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3), # 26x26x16 -> 24x24x32
            nn.ReLU(),
            nn.Flatten()
        )
        self.latent_size = 256
        self.mean_layer = nn.Linear(24*24*32, self.latent_size)
        self.log_var_layer = nn.Linear(24*24*32, self.latent_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 24*24*32),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32, 24, 24)),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3),
            nn.Sigmoid()
        )
        self.BCELoss = nn.BCELoss(reduction='mean')
        self.beta = 1

    def sampler(self,mean,log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return mean + std * eps

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.sampler(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var
    
    def encode(self,x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        return mean, log_var

    def decode(self,z):
        x = self.decoder(z)
        return x

    def vae_loss(self, x_hat, x, mean, log_var):

        # loss function
        kl_divergence = 0.5 * torch.sum(-1 - log_var + mean.pow(2) + log_var.exp())
        #loss = F.mse_loss(x_hat, x, reduction='none') + kl_divergence
        loss = F.binary_cross_entropy(x_hat, x, size_average=False) + kl_divergence

        return loss, loss - kl_divergence, kl_divergence

#Implement the Baseline AutoEncoder for Equations following Author's Implementation:
class EqnAE(nn.Module):
  def __init__(self, charset_length, max_length, latent_rep_size=10, hypers=None):
        super(EqnAE, self).__init__()
        if hypers is None:
            hypers = {'hidden': 100, 'dense': 100, 'conv1': 2, 'conv2': 3, 'conv3': 4}
        # Convolutional layers
        self.conv1 = nn.Conv1d(max_length, hypers['conv1'], kernel_size=hypers['conv1'])
        self.conv2 = nn.Conv1d(hypers['conv1'], hypers['conv2'], kernel_size=hypers['conv2'])
        self.conv3 = nn.Conv1d(hypers['conv2'], hypers['conv3'], kernel_size=hypers['conv3'])

        self.bn1 = nn.BatchNorm1d(hypers['conv1'])
        self.bn2 = nn.BatchNorm1d(hypers['conv2'])
        self.bn3 = nn.BatchNorm1d(hypers['conv3'])
        self.bn4 = nn.BatchNorm1d(latent_rep_size)

        self.fc1 = nn.Linear(40, hypers['dense'])
        self.fc_latent = nn.Linear(hypers['dense'], latent_rep_size)

        self.rev_latent = nn.Linear(latent_rep_size,hypers['dense'])
        # GRU layers
        self.gru1 = nn.GRU(hypers['dense'], hypers['hidden'], batch_first=True)
        self.gru2 = nn.GRU(hypers['hidden'], hypers['hidden'], batch_first=True)
        self.gru3 = nn.GRU(hypers['hidden'], hypers['hidden'], batch_first=True)
        self.fc_final = nn.Linear(hypers['hidden'], charset_length)
        self.time_distributed = nn.Linear(hypers['hidden'], charset_length)

        self.hypers = hypers
        self.max_length = max_length
        self.latent_rep_size = latent_rep_size
        self.charset_length = charset_length
        self.softmax = nn.Softmax(dim=1)

  def encode(self, x):
    #print(x.shape)
    h = F.relu(self.bn1(self.conv1(x)))
    # print(f'After first conv layer: {h.shape}')
    h = F.relu(self.bn2(self.conv2(h)))
    # print(f'After 2 conv layer: {h.shape}')
    h = F.relu(self.bn3(self.conv3(h)))
    # print(f'After 3 conv layer: {h.shape}')
    h = h.view(h.size(0), -1)
    # print(f'After reshape layer: {h.shape}')
    h = F.relu(self.fc1(h))
    # print(f'After  linear layer: {h.shape}')
    z = self.fc_latent(h)
    # print(f'After  mean, log_var layer: {z.shape}')
    return z


  def decode(self, z):
    h = self.bn4(z)
    h = self.rev_latent(h)
    h = h.unsqueeze(1).repeat(1, self.max_length, 1)
    # print(f'After Repeat : {h.shape}')
    h, _ = self.gru1(h)
    h, _ = self.gru2(h)
    h, _ = self.gru3(h)
    decoded = torch.stack([self.time_distributed(h_) for h_ in h], dim=1)
    return self.softmax(decoded.transpose(0,1))

  def forward(self, x):
    z = self.encode(x)
    x_hat = self.decode(z)
    return x_hat

#Implement the Baseline AutoEncoder for Equations following Author's Implementation:
class EqnVAE(nn.Module):
  def __init__(self, charset_length, max_length, latent_rep_size=10, hypers=None):
        super(EqnVAE, self).__init__()
        if hypers is None:
            hypers = {'hidden': 100, 'dense': 100, 'conv1': 2, 'conv2': 3, 'conv3': 4}
        # Convolutional layers
        self.conv1 = nn.Conv1d(max_length, hypers['conv1'], kernel_size=hypers['conv1'])
        self.conv2 = nn.Conv1d(hypers['conv1'], hypers['conv2'], kernel_size=hypers['conv2'])
        self.conv3 = nn.Conv1d(hypers['conv2'], hypers['conv3'], kernel_size=hypers['conv3'])

        self.bn1 = nn.BatchNorm1d(hypers['conv1'])
        self.bn2 = nn.BatchNorm1d(hypers['conv2'])
        self.bn3 = nn.BatchNorm1d(hypers['conv3'])
        self.bn4 = nn.BatchNorm1d(latent_rep_size)

        self.fc1 = nn.Linear(40, hypers['dense'])
        self.fc_mean = nn.Linear(hypers['dense'], latent_rep_size)
        self.fc_logvar = nn.Linear(hypers['dense'], latent_rep_size)

        self.rev_latent = nn.Linear(latent_rep_size,hypers['dense'])
        # GRU layers
        self.gru1 = nn.GRU(hypers['dense'], hypers['hidden'], batch_first=True)
        self.gru2 = nn.GRU(hypers['hidden'], hypers['hidden'], batch_first=True)
        self.gru3 = nn.GRU(hypers['hidden'], hypers['hidden'], batch_first=True)
        self.fc_final = nn.Linear(hypers['hidden'], charset_length)
        self.time_distributed = nn.Linear(hypers['hidden'], charset_length)

        self.hypers = hypers
        self.max_length = max_length
        self.latent_rep_size = latent_rep_size
        self.charset_length = charset_length
        self.softmax = nn.Softmax(dim=1)

  def encode(self, x):
    #print(x.shape)
    h = F.relu(self.bn1(self.conv1(x)))
    # print(f'After first conv layer: {h.shape}')
    h = F.relu(self.bn2(self.conv2(h)))
    # print(f'After 2 conv layer: {h.shape}')
    h = F.relu(self.bn3(self.conv3(h)))
    # print(f'After 3 conv layer: {h.shape}')
    h = h.view(h.size(0), -1)
    # print(f'After reshape layer: {h.shape}')
    h = F.relu(self.fc1(h))
    # print(f'After  linear layer: {h.shape}')
    mean, log_var = self.fc_mean(h), self.fc_logvar(h)
    # print(f'After  mean, log_var layer: {z.shape}')
    return mean, log_var

  def sampler(self,mean,log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.rand_like(std)
    return mean + std * eps

  def decode(self, z):
    h = self.bn4(z)
    h = self.rev_latent(h)
    h = h.unsqueeze(1).repeat(1, self.max_length, 1)
    # print(f'After Repeat : {h.shape}')
    h, _ = self.gru1(h)
    h, _ = self.gru2(h)
    h, _ = self.gru3(h)
    decoded = torch.stack([self.time_distributed(h_) for h_ in h], dim=1)
    return self.softmax(decoded.transpose(0,1))

  def forward(self, x):
    mean, log_var = self.encode(x)
    z = self.sampler(mean, log_var)
    x_hat = self.decode(z)
    return x_hat, mean, log_var

  def vae_loss(self, x_hat, x, mean, log_var):
    # loss function
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    #loss = F.mse_loss(x_hat, x, reduction='mean') + kl_divergence
    loss = self.max_length * F.binary_cross_entropy(x_hat, x, reduction="sum") + kl_divergence
    #loss = customLoss(x,x_hat) + kl_divergence
    return loss, loss - kl_divergence, kl_divergence

