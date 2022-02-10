import torch
from torch.utils.data import DataLoader
from LeNet_5 import *
import torchvision


# test

# 1.Create model
model = LeNet_5()

# 2.Ready Dataset
test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)
# 3.Length
test_dataset_size = len(test_dataset)
print("the test dataset size is {}".format(test_dataset_size))

# 4.DataLoader
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64)

# 5. Set some parameters for testing the network
total_accuracy = 0

# test
model.eval()
with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        model_load = torch.load("model_save/model_62500.pth")
        model.load_state_dict(model_load)
        outputs = model(imgs)
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy = total_accuracy + accuracy
        accuracy = total_accuracy / test_dataset_size
    print("the total accuracy is {}".format(accuracy))
