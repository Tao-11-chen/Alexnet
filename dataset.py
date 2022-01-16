import pathlib
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
warnings.filterwarnings(action='ignore', category=UserWarning)
from torchvision import transforms
from PIL import Image


class makedataset(Dataset):
    def __init__(self, resize, mode):
        super(makedataset, self).__init__()
        self.resize = resize
        self.image, self.label = self.load_data()

        if mode == 'train':
            self.image = self.image[:int(0.9 * len(self.image))]
            self.label = self.label[:int(0.9 * len(self.label))]
        elif mode == 'val':
            self.image = self.image[int(0.9 * len(self.image)):]
            self.label = self.label[int(0.9 * len(self.label)):]
        else:
            self.image = self.image
            self.label = self.label

    def load_data(self):
        image = []
        label = []
        data_root_cat = pathlib.Path('train/cat')
        data_root_dog = pathlib.Path('train/dog')
        for items in data_root_dog.iterdir():
            image.append(items)
            label.append(0)
        for items in data_root_cat.iterdir():
            image.append(items)
            label.append(1)
        image = np.array(image)
        label = np.array(label)
        index = [i for i in range(len(image))]
        np.random.shuffle(index)
        image = image[index]
        label = label[index]
        image = image.tolist()
        label = label.tolist()
        return image, label

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        train_transform = transforms.Compose([transforms.Resize((self.resize, self.resize)),
                                              transforms.ToTensor(),
                                             normalize])
        #image_tensor = ImageFolder(self.image[idx], transform=train_transform)
        image_tensor = train_transform(Image.open(self.image[idx]))
        label_tensor = torch.tensor(self.label[idx])
        return image_tensor, label_tensor







