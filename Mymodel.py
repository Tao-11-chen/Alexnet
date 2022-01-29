import torch.nn as nn
from torch.nn import functional
# model structure of AlexNet


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()    # should inherit torch.nn.Module
        self.c1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2)
        # Conv1: input: RGB 3channels;output: 48channels;1kernel_size 11;stride 4,padding 2
        self.ReLU = nn.ReLU()
        # Activation function1
        self.c2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        # Conv2: input: 45channels;output: 128channels;kernel_size 5;stride 1,padding 2
        self.s2 = nn.MaxPool2d(2)
        # down-sampling using MaxPool 1
        self.c3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        # Conv3: input: 128channels;output: 192channels;kernel_size 3;stride 1,padding 1
        self.s3 = nn.MaxPool2d(2)
        # down-sampling using MaxPool 2
        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        # Conv4: input: 192channels;output: 192channels;kernel_size 3;stride 1,padding 1
        self.c5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        # Conv5: input: 192channels;output: 128channels;kernel_size 3;stride 1,padding 1
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # down-sampling using MaxPool 3
        self.flatten = nn.Flatten()
        # demensionality reduction
        self.f6 = nn.Linear(4608, 2048)
        # fully-connected
        self.f7 = nn.Linear(2048, 2048)
        # fully-connected
        self.f8 = nn.Linear(2048, 1000)
        # fully-connected
        self.f9 = nn.Linear(1000, 2)
        # finally output two classes

    def forward(self, x):
        # connect the parts above using
        x = self.ReLU(self.c1(x))
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s3(x)
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = functional.dropout(x, p=0.5)    # randomly drop some data to improve the generalization ability.
        x = self.f7(x)
        x = functional.dropout(x, p=0.5)    # randomly drop some data to improve the generalization ability.
        x = self.f8(x)
        x = functional.dropout(x, p=0.5)    # randomly drop some data to improve the generalization ability.
        x = self.f9(x)
        return x

