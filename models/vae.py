import datetime
import math
import time

import cv2
import torch
from torch.utils.data import DataLoader

from face_orientation.face_orientation_ai import OrientationLoader

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train():
    img_size = 28
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
    ])

    # dataset = OrientationLoader(transform=trans, device=device)
    dataset = torch.load("./preprocessed.pth")
    batch_size = 128
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    latent_dims = 100

    vae = VariationalAutoencoder(latent_dims).to(device)  # GPU
    try:
        vae.load_state_dict(torch.load("./vae.pth"))
        pass
    except Exception as e:
        print(e)
    vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0001)

    plt.ion()
    plt.figure(figsize=(4, 1))
    losses = []
    for epoch in range(10000):
        running_loss = 0.0
        for i, batch in enumerate(loader):
            # get the training inputs
            images = batch["image"]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = vae(images)
            loss = ((images - outputs)**2).sum() + vae.encoder.kl
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                loss_ = running_loss / 100
                print(f"Epoch: {epoch + 1}, Run: {i+1 :5d}, Loss: {loss_ :.5f}")
                running_loss = 0.0

        losses.append(loss_)
        if epoch % 10 == 9:
            torch.save(vae.state_dict(), "./vae.pth")
            img = torchvision.utils.make_grid(torch.cat([images, outputs]), nrow=int(math.sqrt(2*batch_size)))
            cv2.imshow("progress", img.permute(1, 2, 0).detach().cpu().numpy())
            cv2.waitKey(1)
            torchvision.utils.save_image(img, f"VAE.jpg")
            plt.clf()
            plt.plot(losses)
            plt.pause(0.005)
        if epoch % 250 == 249:
            torch.save(vae.state_dict(), f"./bckup/vae_{epoch}_{time.strftime('%Y%m%d-%H%M%S')}.pth")

    print("Finished")


def preprocess():
    img_size = 28
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
    ])

    dataset = OrientationLoader(transform=trans, device=device)
    l = [d for d in dataset]
    torch.save(l, "./preprocessed.pth")


if __name__ == '__main__':
    start = datetime.datetime.now()
    train()
    print(f"Took {datetime.datetime.now() - start}")
