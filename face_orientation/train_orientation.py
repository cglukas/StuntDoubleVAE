import datetime

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader

from face_orientation.face_orientation_ai import OrientationLoader
from models.network import ClassifierNetLarge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = 128
trans = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((img_size, img_size)),
    torchvision.transforms.ToTensor(),
])

def train():
    dataset = OrientationLoader(transform=trans, device=device)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    myNet = ClassifierNetLarge()
    myNet.load_state_dict(torch.load("./full_set.pth"))
    myNet.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(myNet.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1000):
        running_loss = 0.0
        for i, batch in enumerate(loader):
            # get the training inputs
            images = batch["image"]
            labels = batch["orientation"]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = myNet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f"Epoch: {epoch + 1}, Run: {i+1 :5d}, Loss: {running_loss/200:.5f}")
                running_loss = 0.0

        torch.save(myNet.state_dict(), "./full_set.pth")

    print("Finished")

def test():
    device = torch.device("cpu")

    dataset = OrientationLoader(transform=trans, device=device)
    loader = iter(DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4))
    data = next(loader)
    images, labels = data["image"], data["orientation"]
    with torch.no_grad():
        myNet = ClassifierNetLarge()
        myNet.load_state_dict(torch.load("./test2.pth"))
        myNet.to(device)
        myNet.eval()
        print(labels)
        predicted = myNet(images)

    _, a = plt.subplots(2,2)
    n_lables = len(labels)
    colors= plt.cm.get_cmap("hsv", n_lables+1)

    a =a.flatten()
    for i, ax in enumerate(a):
        ax.imshow(images[i].permute(1,2,0), cmap="gray" )

    plt.figure()
    ax = plt.axes(projection='3d')

    for i in range(n_lables):

        x1, y1, z1 = labels[i]
        x2, y2, z2 = predicted[i]

        ax.plot3D([0, x1], [0, y1], [0, z1], c=colors(i))
        ax.plot3D([0, x2], [0, y2], [0, z2], c=colors(i))
    plt.show()


if __name__ == '__main__':
    start = datetime.datetime.now()
    train()
    print(f"Took {datetime.datetime.now() - start}")
