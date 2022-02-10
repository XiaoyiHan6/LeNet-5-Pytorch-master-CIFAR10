from torch.utils.data import DataLoader
from LeNet_5 import *
import torchvision
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 1. torch choose cuda or cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 2.Create SummaryWriter
writer = SummaryWriter("log_loss")

# 3.Ready dataset
train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)

# 4.Length
train_dataset_size = len(train_dataset)
print("the train dataset size is {}".format(train_dataset_size))

# 5.DataLoader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64)

# 6.Create model
model = LeNet_5()
# a.add cuda
model = model.to(device=device)

# 7.Create loss
cross_entropy_loss = nn.CrossEntropyLoss()
# b.add cuda
cross_entropy_loss = cross_entropy_loss.to(device=device)

# 8.Optimizer
learning_rate = 1e-2
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 9. Set some parameters to control loop
# epoch
epoch = 80

total_train_step = 0

for i in range(epoch):
    print(" -----------------the {} number of training epoch --------------".format(i + 1))
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss_train = cross_entropy_loss(outputs, targets)

        optim.zero_grad()
        loss_train.backward()
        optim.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("the training step is {} and its loss of model is {}".format(total_train_step, loss_train.item()))
            writer.add_scalar("train_loss", loss_train.item(), total_train_step)
            if total_train_step % 10000 == 0:
                torch.save(model.state_dict(), "model_save/model_{}_GPU.pth".format(total_train_step))
                print("the model of {} training step was saved! ".format(total_train_step))
            if i == (epoch - 1):
                torch.save(model.state_dict(), "model_save/model_{}_GPU.pth".format(total_train_step))
                print("the model of {} training step was saved! ".format(total_train_step))
writer.close()
