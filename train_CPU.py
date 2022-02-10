from torch.utils.data import DataLoader
from LeNet_5 import *
import torchvision
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 1.Create SummaryWriter
writer = SummaryWriter("log_loss")

# 2.Ready dataset
train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)

# 3.Length
train_dataset_size = len(train_dataset)
print("the train dataset size is {}".format(train_dataset_size))

# 4.DataLoader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64)

# 5.Create model
model = LeNet_5()

# 6.Create loss
cross_entropy_loss = nn.CrossEntropyLoss()

# 7.Optimizer
learning_rate = 1e-3
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 8. Set some parameters to control loop
# epoch
epoch = 200

total_train_step = 0

for i in range(epoch):
    print(" -----------------the {} number of training epoch --------------".format(i + 1))
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        print(imgs.shape)
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
                torch.save(model.state_dict(), "model_save/model_{}.pth".format(total_train_step))
                print("the model of {} training step was saved! ".format(total_train_step))
            if i == (epoch - 1):
                torch.save(model.state_dict(), "model_save/model_{}.pth".format(total_train_step))
                print("the model of {} training step was saved! ".format(total_train_step))
writer.close()
