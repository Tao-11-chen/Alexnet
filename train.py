import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from dataset import makedataset
from Mymodel import AlexNet

train_db = makedataset(224, mode='train')
val_db = makedataset(224, mode='val')
test_db = makedataset(224, mode='test')
train_loader = DataLoader(train_db, batch_size=32, shuffle=True,num_workers=8)
val_loader = DataLoader(val_db, batch_size=32, shuffle=True,num_workers=8)
test_loader = DataLoader(test_db, batch_size=32, shuffle=True,num_workers=8)
#print('num_train', len(train_loader.dataset))
#print('num_val', len(val_loader.dataset))
#print('num_test', len(test_loader.dataset))
device = torch.device('cuda')

def compute_acc(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        _, pred = torch.max(output, axis=1)
        correct += pred.eq(y).sum()
    return correct/total

model = AlexNet().to(device)
optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
loss_f = CrossEntropyLoss()

def train():
    loss_list = []
    epoch_list = []
    acc_list = []
    best_val_acc, best_epoch = 0, 0
    for epoch in range(20):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            model.train()
            pred = model(x)
            loss = loss_f(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = compute_acc(model, val_loader)
        epoch_list.append(epoch)
        loss_list.append(loss.float())
        acc_list.append(val_acc.__float__())
        print('Epoch:', epoch, 'Loss:', loss, 'Val_acc', val_acc)
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), 'Weight.mdl')
            best_val_acc = val_acc
            best_epoch = epoch
    print('Best_val_acc:', best_val_acc,'Best_epoch', best_epoch)
    return acc_list,epoch_list,loss_list

def test():
    model.load_state_dict(torch.load('Weight.mdl'))
    print('Test_acc:', compute_acc(model, test_loader))

if __name__ == "__main__":
    acc_list, epoch_list, loss_list = train()
    test()
    acc_list = torch.tensor(acc_list,device='cpu')
    epoch_list = torch.tensor(epoch_list,device='cpu')
    loss_list = torch.tensor(loss_list,device='cpu')
    plt.figure()
    plt.plot(epoch_list, loss_list, color='green', label='loss')
    plt.plot(epoch_list, acc_list, color='red', label='training accuracy')
    plt.legend()
    plt.savefig('pci.png')
    plt.show()
