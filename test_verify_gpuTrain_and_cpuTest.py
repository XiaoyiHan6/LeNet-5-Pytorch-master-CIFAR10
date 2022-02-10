import torch
import cv2
import torchvision

from LeNet_5 import *

# test

# 1.Create model
model = LeNet_5()

# 2.Ready Data
img = cv2.imread("dog.jpg")

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize((32, 32))])
img = transform(img)
img = img.reshape(1, 3, 32, 32)

# test
model.eval()
model_load = torch.load("model_save/model_62500_GPU.pth", map_location=torch.device("cpu"))
model.load_state_dict(model_load)
with torch.no_grad():
    output = model(img)
    print(output)
    cls = output.argmax(1)

    print("the classification of object is {}".format(cls))
