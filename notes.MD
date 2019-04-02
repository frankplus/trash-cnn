## Differences from trashnet model
Here are listed some differences with our implementation and the architecture of trashnet by Mindy Yang and Gary Thung

#### optimization options
- learning rate: not set ; trashnet=1.25e-5
- learning decay factor (newLR = oldLR * <lrDecayFactor>): not set ; trashnet=0.9
- weight decay: not set ; trashnet=2.5e-2
- weight initialization method: not set ; trashnet=kaiming

#### data augmentation
trashnet for the training set selects randomly between: random crop, horizontal flip, add noise.\
We are randomly augmenting with: shear_range=0.2, zoom_range=0.2, horizontal_flip=True

#### activation function for fully connected layer
thrashnet is using [this](https://pytorch.org/docs/stable/nn.html#threshold) while we are using ReLU with some negative slope

## Using transfer learning
Following [this paper](http://cs230.stanford.edu/projects_spring_2018/reports/8290808.pdf) we tried to use transfer 
learning using the VGG-19 architecture pretrained with imagenet. On a first try with few epochs it seems to perform 
better than the model based on trashnet architecture.
Here are the results with 3 epochs:
```
903s 13s/step - loss: 1.6820 - acc: 0.3509 - val_loss: 1.2836 - val_acc: 0.5495
909s 14s/step - loss: 1.3316 - acc: 0.4778 - val_loss: 1.1821 - val_acc: 0.5286
894s 13s/step - loss: 1.2114 - acc: 0.5072 - val_loss: 1.1839 - val_acc: 0.5078
```
Validation accuracy is higher than training set accuracy because on the training set we are doing data augmentation by
shear, zoom, horizontal flip, rotation and shift. \
Notice that while the training accuracy improves, the validation accuracy does not. I think that it suffers from 
overfitting, further studies are needed.
TODO: 
- increase epochs
- try with leakyReLU activation function
- add regularization
- test different epsilon and learning rate 