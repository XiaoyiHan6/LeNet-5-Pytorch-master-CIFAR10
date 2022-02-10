import torch.optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

# Create SummaryWriter of tensorboard
writer = SummaryWriter("log_loss")


# Ready model
class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.model = nn.Sequential(
            # input:3@32x32
            # 6@28x28
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0, stride=1),
            # 6@14x14
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # 16@10x10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1),
            # 16@5x5
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),

        )

    def forward(self, x):
        x = self.model(x)
        return x


# Ready dataset
train_dataset = torchvision.datasets.CIFAR10(root="data", train=True,
                                             transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root="data", train=False,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True)
# Length
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print("the length of train_datase is {}".format(train_dataset_size))
print("the lenght of test_dataset is {}".format(test_dataset_size))

# DataLoader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64)

# Create Model
model = LeNet_5()

# Create Loss
cross_entropy_loss = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 1e-2
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Set some parameters for training the network
# Record the number of train
total_train_step = 0

# Record the number of test
total_test_step = 0

# epoch
epoch = 10

for i in range(epoch):
    print("------------ the {} training start----------".format(i + 1))
    # train
    for data in train_dataloader:
        imgs, targets = data
        output = model(imgs)
        loss_train = cross_entropy_loss(output, targets)

        # optim
        optim.zero_grad()
        loss_train.backward()
        optim.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("the number of training is {} and its loss is {}".format(total_train_step, loss_train.item()))
            # Tensorboard
            writer.add_scalar("train_loss", loss_train.item(), total_train_step)

    # test
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = model(imgs)
            loss_test = cross_entropy_loss(outputs, targets)
            total_test_loss = total_test_loss + loss_test
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
        total_test_step = total_test_step + 1
        print("total_test_step: ", total_test_step)
        print("the total testing loss is {} ".format(total_test_loss.item()))
        print("the total accuracy is {}".format(total_accuracy / test_dataset_size))
        # Tensorboard
        writer.add_scalar("test_accuracy", total_accuracy / test_dataset_size, total_test_step)

# save model
torch.save(model.state_dict(), "model_{}.pth".format(i))
print("the model was saved! ")
# Summary close
writer.close()
