# LeNet-5-Pytorch-master-CIFAR10
Hello World! This is my first code for GitHub! And It's called LeNet-5-Pytorch-master-CIFAR10 by me! Although this code is so simple, I wrote it seriously!

And, there are some my Chinese communication websites such as CSDN and Quora(Chinese) where I explain this code. CSDN:https://blog.csdn.net/XiaoyYidiaodiao/article/details/122720320?spm=1001.2014.3001.5501
Quora(Chinese)-Zhihu:https://zhuanlan.zhihu.com/p/463827403

We can see from the project directory above that our project can use both GPU training models and CPU training models.
And, we can test model by GPU and CPU, or GPU training and CPU testing.

|File Name   |      explanation      |      PS       | 
|------------|:---------------------:|---------------|
|train_CPU.py  | CPU training models |                                                                 |
|train_GPU.py  | GPU training models | device = torch.device("cuda") , model = model.to(device=device) |
|train_GPU_1.py| GPU training models |                        model = model.cuda()                     |
|test_accuracy_CPU.py              |     CPU testing models                |                   calculate    accuracy                         |
|test_accuracy_GPU.py              |     GPU testing models                |                   calculate    accuracy                         |
|test_accuracy_gpuTrain_and cpuTest.py              |  GPU training models and CPU testing models |      calculate     accuracy              |
|test_verify_CPU.py              |   CPU testing models |     classification             |
|test_verify_gpuTrain_and_cpuTest.py|   GPU training models and CPU testing models |     classification             |

If we want to run xx.py, we will use the following code:

> python xx.py

for example:

we choose the following picture to test

![image](https://user-images.githubusercontent.com/98302212/153441623-267d9742-c09c-4006-9eba-2cb117cb1543.png)


1. we want to run test_verify_gpuTrain_and_cpuTest.py

![image](https://user-images.githubusercontent.com/98302212/153442228-f52c5575-f318-46ef-897f-72dc76af93c8.png)


![image](https://user-images.githubusercontent.com/98302212/153442716-106457eb-d009-4292-9582-7101e824f433.png)

So, we successed!

2. we want to run test_accuracy_gpuTrain_and_cpuTest.py

![image](https://user-images.githubusercontent.com/98302212/153443218-8bb5c8e2-f878-4e73-83dc-2ad9f5af03bf.png)

So, we successed!
