import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
Implementation of Autoencoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim//2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )
    def forward(self, x):
        #TODO: 5%
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        raise NotImplementedError
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 5%
        # initialize the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_func = nn.MSELoss()
        loos_arr = []
        for epoch in range(epochs):
            start = 0
            total_loss = 0
            while(start < X.shape[0]):
                bsize = min(batch_size, X.shape[0] - start)
                X_batch = X[start:start + bsize - 1]
                start += bsize
                # forward the batch
                X_batch = torch.tensor(X_batch, dtype=torch.float32)
                output = self.forward(X_batch)
                # calculate the loss
                loss = loss_func(output, X_batch)
                total_loss += loss.item()
                optimizer.zero_grad()
                # backward the loss
                loss.backward()
                # update the weights
                optimizer.step()
            total_loss = total_loss / (X.shape[0] / batch_size)
            loos_arr.append(total_loss)
        # plot the shit
        return
        raise NotImplementedError
    
    def transform(self, X):
        #TODO: 2%
        X_ten = torch.tensor(X, dtype=torch.float32)
        ret = self.encoder(X_ten)
        ret = ret.detach().numpy()
        return ret
        raise NotImplementedError
    
    def reconstruct(self, X):
        #TODO: 2%
        X_ten = X
        ret = self.forward(X_ten)
        ret = ret.detach().numpy()
        return ret
        raise NotImplementedError


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim,encoding_dim)
        self.noise_factor = noise_factor
    
    def add_noise(self, x):
        #TODO: 3%
        noise = torch.zeros([1,x.shape[1]], dtype=torch.float32)
        for i in range(x.shape[1]):
            noise[0,i] = self.noise_factor * torch.randn(1)
        return x + noise
        raise NotImplementedError
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 4%
        # initialize the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_func = nn.MSELoss()
        loos_arr = []
        for epoch in range(epochs):
            start = 0
            total_loss = 0
            while(start < X.shape[0]):
                bsize = min(batch_size, X.shape[0] - start)
                X_batch = X[start:start + bsize - 1]
                start += bsize
                # forward the batch
                X_batch = torch.tensor(X_batch, dtype=torch.float32)
                X_batch = self.add_noise(X_batch)
                output = self.forward(X_batch)
                # calculate the loss
                loss = loss_func(output, X_batch)
                total_loss += loss.item()
                optimizer.zero_grad()
                # backward the loss
                loss.backward()
                # update the weights
                optimizer.step()
            total_loss = total_loss / (X.shape[0] / batch_size)
            loos_arr.append(total_loss)
        # plot the shit
        return
        raise NotImplementedError
