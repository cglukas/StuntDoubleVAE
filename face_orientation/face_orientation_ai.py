from pathlib import Path
from typing import Dict

import PIL.Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import pickle
from matplotlib import pyplot


FACE_ORIENTATION_DATA = r"C:\Users\Lukas\PycharmProjects\DeepFaceLab\FaceSets\full_set.dat"

class OrientationLoader(Dataset):

    def __init__(self, transform=None, device="cpu"):
        """

        Args:
            transform: transforms to apply to the image before loading
        """
        self.folder = Path(r"C:\Users\Lukas\PycharmProjects\DeepFaceLab\FaceSets\full_set")
        with open(FACE_ORIENTATION_DATA, "rb") as file:
            self.orientations: Dict = pickle.load(file)
        self.name_list = list(self.orientations.keys())
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.orientations.keys())

    def __getitem__(self, index) -> Dict:
        if torch.is_tensor(index):
            index = index.tolist()

        name = self.name_list[index]
        orientation = self.orientations[name]
        orientation = torch.tensor(orientation, dtype=torch.float32, device=self.device)
        image_name = self.folder / name
        with open(image_name, "rb") as file:
            image = PIL.Image.open(file).convert("RGB")

        if self.transform:
            image = self.transform(image)
            image = image.to(self.device)

        sample = {"image": image, "orientation": orientation}
        return sample


def show_image(image: torch.Tensor):
    """Show an image tensor as pyplot

    Args:
        image: tensor of image (width, height, channels)
    """
    permute = image.permute(1, 2, 0)
    pyplot.imshow(permute)
    pyplot.show()


def main():
    img_size = 256
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
    ])
    dataset = OrientationLoader(transform=trans)

    # for index, sample in enumerate(dataset):
    #     image: torch.Tensor = sample["image"]
    #     print(image.shape, sample["orientation"].shape)
    #     show_image(image)
    #     break




if __name__ == '__main__':
    main()