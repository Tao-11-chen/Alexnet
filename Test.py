import torch
import warnings
warnings.filterwarnings(action='ignore')
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from Mymodel import AlexNet
device = torch.device('cuda')


if __name__ == '__main__':
    model = AlexNet().to(device)
    model.load_state_dict(torch.load('Weight.mdl'))
    model.eval()
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor()])
    image_tensor = ImageFolder('test1', transform=train_transform)
    test_loader = DataLoader(image_tensor,batch_size=16,num_workers=8)
    for image, label in test_loader:
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        _, pred = torch.max(output, axis=1)
        print(pred.tolist())
        print(label.tolist())
        print("---------")