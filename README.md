
# Modified ResNet architecture on CIFAR 10 with less than 5 million parameters

In this project, we aimed to design a modified ResNet architecture for the CIFAR10 image classification dataset with no more than 5 million parameters. We utilized PyTorch to implement the ResNet architecture, and our modified ResNet achieved an accuracy of 93%  on the test set. Our methodology involved experimenting with different ResNet architectures and hyperparameters, such as the number of layers and blocks, and evaluating their performance on the CIFAR10 dataset. We found that modifying the original ResNet18 architecture to include 4 layers with respective block size of 2, 1, 1, 1 resulted in the highest test accuracy.


## Models Considered 


| Block Layer Architecture | Epochs | Number of Parameters | Optimizer | Dropout | Learning Rate | Training Accuracy | Testing Accuracy |
|--------------------------|--------|----------------------|-----------|---------|---------------|-------------------|------------------|
| [2,1,1,1]                | 100    | 4977226              | Adadelta  | 0.4     | 0.1           | 99.76%            | 92.36%           |
| [2,1,1,1]                | 100    | 4977226              | SGD       | 0.4     | 0.1           | 98.32%            | 92.40%           |
| [2,1,1,1]                | 150    | 4977226              | SGD       | 0.2     | 0.01          | 99.98%            | 93.82%           |
| [2,2,2]                  | 120    | 2777674              | SGD       | 0.2     | 0.1           | 97.84%            | 92.17%           |
| [3,3,3]                  | 130    | 4327754              | SGD       | 0.4     | 0.1           | 96.72%            | 92.09%           |


While Model3 gets the highest accuracy but it was trained for higher epochs and here the training accuracy peaked at almost 100%, which would be the case for all others but all others were stopped after convergence, and hence even though Model3 achives higher accuracy Model2 generalizes well and doesn't overfit. 

## File Structure 

The file Structure of the repository is as follows:- 

- ```data``` : CIFAR 10 dataset downloaded by pytroch
- ```evaluate-and-plot-confusion.py``` : Python file to test the models as listed in section 3 and 4 and plot the confusion matrix 

- ```Model with Adadelta-2111-Dropout-0.4``` : Model1 Structure

    - ```architecture.py```: Model Architecture
    - ```DL_Mini_Adadelta_2111_0_4```: Python Notebook for training and evaluating the model, contains the whole codebase for this project for this model 

    - ```model.pt```: The saved dict file for the  model with best testing accuracy during training for this particular model 

    - ```trainTestCurve```: The Train Test Curve for the  model with best testing accuracy during training for this particular model 

- ```Model with SGD-2111-Dropout-0.4```: Model2 Structure, same file structure(different architecture) as Model1

- ```Model with SGD-2111-LR-0.01-Dropout-0.2```: Model3 Structure, same file structure(different architecture) as Model1

- ```Model with SGD-222-Dropout-0.2```: Model4 Structure, same file structure(different architecture) as Model1

- ```Model with SGD-333-Dropout-0.4```: Model5 Structure, same file structure(different architecture) as Model1



## Evaluate Model and Plot Confusion Matrix

To evaluate the model a file name 
```
evaluate-and-plot-confusion.py
```

has been added to the project. This file imports the specified architecture and evaluates the trained model on the Test Dataset. 

Each folder contains a the architecture file called architecture.py and is in the folder for each model. To specifiy which architecture to run test on pass the required system argument according to the syntax and model reference below:- 

```
python evaluate-and-plot-confusion.py arg 
```

Here arg can take the following values with specification of which architecture they represent:- 

```
Model1: Model with Adadelta-2111-Dropout-0.4

Model2: Model with SGD-2111-Dropout-0.4 (Default)

Model3: Model with SGD-2111-LR-0.01-Dropout-0.2

Model4: Model with SGD-222-Dropout-0.2

Model5: Model with SGD-333-Dropout-0.4

```

Example:- 

```
python evaluate-and-plot-confusion.py Model1

```

Model2 is Default because of its high accuracy and robustness. 


## Authors

- [@dipitvasdev](https://www.github.com/dipitvasdev)
- [@palakkeni5] (https://github.com/palakkeni5)


## Acknowledgements

 - [How to import a Python module given the full path?](https://www.geeksforgeeks.org/how-to-import-a-python-module-given-the-full-path/)
 - [PyTorch Imagenet](https://github.com/sanghoon/pytorch_imagenet/blob/master/toy_cifar.py)
 - [Train CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar)

