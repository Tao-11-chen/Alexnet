import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from Dataset import MakeDataset
from Mymodel import AlexNet


# compute model's accuracy
def compute_acc(model, loader):
    model.eval()    # change to eval mode
    correct = 0     # init number of correct
    total = len(loader.dataset)    # length of dataset
    for x, y in loader:            # traverse DataLoader and put it into GPU
        x = x.to(device)
        y = y.to(device)
        output = model(x)          # make output
        _, pred = torch.max(output, axis=1)    # if predict == label
        correct += pred.eq(y).sum()    # calculate accuracy
    return correct/total


def train():
    loss_list = []
    epoch_list = []
    acc_list = []
    best_val_acc, best_epoch = 0, 0
    for epoch in range(20):      # 20 epoch at a time
        for x, y in train_loader:    # traverse DataLoader and put it into GPU
            x = x.to(device)
            y = y.to(device)
            model.train()
            pred = model(x)          # output predict
            loss = loss_f(pred, y)     # calculate loss
            # -----------------------------------
            # three steps to train
            optimizer.zero_grad()
            # 1. set grad to zero
            loss.backward()
            # 2. loss backward
            optimizer.step()
            # 3. start to update grad
            # -----------------------------------
        val_acc = compute_acc(model, val_loader)     # calculate accurate
        print('Epoch:', epoch, 'Loss:', loss, 'Val_acc', val_acc)  # print information
        # -----------------------------
        # put data to the lists to draw picture
        epoch_list.append(epoch)
        loss_list.append(loss.float())
        acc_list.append(val_acc.__float__())
        # -----------------------------
        # save the epoch which has the best accuracy
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), 'Weight.mdl')
            best_val_acc = val_acc
            best_epoch = epoch
        # ------------------------------
    # print the best of the 20 epochs and return lists
    print('Best_val_acc:', best_val_acc, 'Best_epoch', best_epoch)
    return acc_list, epoch_list, loss_list


def test():   # test the trained model
    model.load_state_dict(torch.load('Weight.mdl'))
    print('Test_acc:', compute_acc(model, test_loader))


if __name__ == "__main__":
    if torch.cuda.is_available(): # use cuda to train if cuda available
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # ------------------------------------------
    # use MakeDataset class and torch.utils.Data.DataLoader to load the data
    train_db = MakeDataset(224, mode='train')
    val_db = MakeDataset(224, mode='val')
    test_db = MakeDataset(224, mode='test')
    train_loader = DataLoader(train_db, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_db, batch_size=32, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_db, batch_size=32, shuffle=True, num_workers=8)
    # ------------------------------------------

    model = AlexNet().to(device)     # put the model into cuda core
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # setting optimizer,using SGD method,learning rate = 0.01
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
    # let learning rate *0.25 every 10 steps
    loss_f = CrossEntropyLoss()
    # define loss function: using log loss

    acc_list, epoch_list, loss_list = train()   # train
    test()                                      # test

    # draw the picture of loss and epoch and accuracy
    acc_list = torch.tensor(acc_list, device='cpu')
    epoch_list = torch.tensor(epoch_list, device='cpu')
    loss_list = torch.tensor(loss_list, device='cpu')
    plt.figure()
    plt.plot(epoch_list, loss_list, color='green', label='loss')
    plt.plot(epoch_list, acc_list, color='red', label='training accuracy')
    plt.legend()
    plt.savefig('pci.png')
    plt.show()
