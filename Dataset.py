import pathlib
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
warnings.filterwarnings(action='ignore', category=UserWarning)   # ignore an unknown warning in my torch library
from torchvision import transforms
from PIL import Image

# this file is used to make our own datasets


class MakeDataset(Dataset):
    def __init__(self, resize, mode):
        super(MakeDataset, self).__init__()     # should inherit torch.utils.data.Dataset
        self.resize = resize                 # resize to the same format(conv requires)
        self.image, self.label = self.load_data()    # define image and label list

        # split the dataset for train and eval,9:1
        # uses the whole dataset to do the last test
        if mode == 'train':
            self.image = self.image[:int(0.9 * len(self.image))]
            self.label = self.label[:int(0.9 * len(self.label))]
        elif mode == 'val':
            self.image = self.image[int(0.9 * len(self.image)):]
            self.label = self.label[int(0.9 * len(self.label)):]
        else:   # test
            self.image = self.image
            self.label = self.label

    def load_data(self):      # load data from the picture source (Using pathlib)
        image = []
        label = []
        data_root_cat = pathlib.Path('train/cat')
        data_root_dog = pathlib.Path('train/dog')
        for items in data_root_dog.iterdir():        # put dog data to the list
            image.append(items)
            label.append(0)
        for items in data_root_cat.iterdir():        # put cat data to the list
            image.append(items)
            label.append(1)
        # ---------------------------------------------
        # upset the data
        # (very important because when the model learns too many same type continuously,may loss generalization ability)
        image = np.array(image)
        label = np.array(label)
        # 1. uses numpy to transform to nparrary
        index = [i for i in range(len(image))]
        np.random.shuffle(index)
        # 2. use random to generate a random list
        image = image[index]
        label = label[index]
        # 3. refer to the random list  and shuffle the nparray
        image = image.tolist()
        label = label.tolist()
        # 4. transform the nparrary back to python list
        # ---------------------------------------------
        return image, label

    def __len__(self):          # not used but should be overwrite cus super classes have this
        return len(self.image)

    # return the asked image one by one
    # important cus some pretreatment on the image should be done here
    # (Note that on the code above, the image list only contains the path of each image)
    def __getitem__(self, idx):
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # using torchvision.transforms
        train_transform = transforms.Compose([transforms.Resize((self.resize, self.resize)),
                                              transforms.ToTensor(),
                                              normalize])
        # the compose contains three steps:
        # 1. resize the image(source pictures usually has different sizes)
        # 2. turn the picture to tensor(which CNN needs)
        # 3. normalize the image for CNN to extract features more conveniently(You'd better do it)
        image_tensor = train_transform(Image.open(self.image[idx]))
        # using PIL.Image to open the image and do the compose
        label_tensor = torch.tensor(self.label[idx])
        # Don't forget to transform the label to tensor too
        return image_tensor, label_tensor







