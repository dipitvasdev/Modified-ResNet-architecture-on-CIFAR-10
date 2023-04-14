"""

Run this file to evaluate and plot confusion matrix for a given architecture, the architecture file is called architecture.py and is in the folder for each model.  


This file can be run simply as follows:- 



```
python evaluate-and-plot-confusion.py arg
```

With arg can take following values:- 

Model1: Model with Adadelta-2111-Dropout-0.4

Model2: Model with SGD-2111-Dropout-0.4 (Default)

Model3: Model with SGD-2111-LR-0.01-Dropout-0.2

Model4: Model with SGD-222-Dropout-0.2

Model5: Model with SGD-333-Dropout-0.4

Example:- 

```
python evaluate-and-plot-confusion.py Model1

```


"""



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import os

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm


import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay



## Load Data 

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')



## Import Correct model according to specifications in arguments

import sys 
import importlib.util


list_arg = sys.argv 

## Model Dictionary:- 


### Ref:- https://www.geeksforgeeks.org/how-to-import-a-python-module-given-the-full-path/
""" 

Model 1: Model with Adadelta-2111-Dropout-0.4

Model 2: Model with SGD-2111-Dropout-0.4 (Default)

Model 3: Model with SGD-2111-LR-0.01-Dropout-0.2

Model 4: Model with SGD-222-Dropout-0.2

Model 5: Model with SGD-333-Dropout-0.4


"""


if(len(list_arg) >= 3):
  raise Exception("Please provide only 1 argument")
elif(list_arg[1] == "Model1"):
  sys.path.insert(0, './Model with Adadelta-2111-Dropout-0.4')
  from architecture import ResNet18
  os.chdir("./Model with Adadelta-2111-Dropout-0.4")
elif(list_arg[1] == "Model2"):
  sys.path.insert(0, './Model with SGD-2111-Dropout-0.4')
  from architecture import ResNet18
  os.chdir("./Model with SGD-2111-Dropout-0.4")
elif(list_arg[1] == "Model3"):
  sys.path.insert(0, './Model with SGD-2111-LR-0.01-Dropout-0.2')
  from architecture import ResNet18
  os.chdir("./Model with SGD-2111-LR-0.01-Dropout-0.2")
elif(list_arg[1] == "Model4"):
  sys.path.insert(0, './Model with SGD-222-Dropout-0.2')
  from architecture import ResNet18
  os.chdir("./Model with SGD-222-Dropout-0.2")
elif(list_arg[1] == "Model5"):
  sys.path.insert(0, './Model with SGD-333-Dropout-0.4')
  from architecture import ResNet18
  os.chdir("./Model with SGD-333-Dropout-0.4")
elif(len(list_arg) == 1):
  sys.path.insert(0, '/content/Model with SGD-2111-Dropout-0.4')
  from architecture import ResNet18
  os.chdir("./Model with SGD-2111-Dropout-0.4")
else:
  raise Exception("Invalid Arguments")





device = 'cuda' if torch.cuda.is_available() else 'cpu'






## Load Model 

net = ResNet18()
net.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
net.to(device)

## Set Parameters

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)


## Get Test Function 

def test(epoch):
    ## Setting models to evaluation mode
    net.eval()
    
    ## Initialize epoch loss and accuracy
    test_loss = 0
    test_acc = 0 

    ## Number of correct examples
    correct = 0
    total = 0


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            ## Move inputs and targets to cuda

            inputs, targets = inputs.to(device), targets.to(device)

            ## Get predictions

            outputs = net(inputs)

            ## Compute Loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            ## Computer Accuracy 
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    test_acc = 100.*correct/total
    
    return test_loss / len(testloader) , test_acc



## Loss and Accuracy 
test_loss, test_acc = test(net)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')

## Confusion Matrix 


def get_predictions(model, iterator, device):
  ## Set model to evaluate mode 

  model.eval()
  
  ## Labels and probability lists 

  labels = []
  probs = []
  
  with torch.no_grad():
    for (x,y) in iterator:
      x = x.to(device)
      y = y.to(device)
      y_pred = model(x)
      y_prob = F.softmax(y_pred, dim = -1)
      top_pred = y_prob.argmax(1, keepdim = True)
      labels.append(y.cpu())
      probs.append(y_prob.cpu())
  labels = torch.cat(labels, dim = 0)
  probs = torch.cat(probs, dim = 0)
  return labels, probs


labels, probs = get_predictions(net, testloader, device)
pred_labels = torch.argmax(probs, 1)


def plot_confusion_matrix(labels, pred_labels, classes):
  fig = plt.figure(figsize = (10, 10));
  ax = fig.add_subplot(1, 1, 1);
  cm = confusion_matrix(labels, pred_labels);
  cm = ConfusionMatrixDisplay(cm, display_labels = classes);
  cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
  plt.xticks(rotation = 20)



plot_confusion_matrix(labels, pred_labels, classes)

os.chdir("/")