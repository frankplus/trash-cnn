## Using transfer learning
Following [this paper](http://cs230.stanford.edu/projects_spring_2018/reports/8290808.pdf) we tried to use transfer 
learning using the VGG-19 architecture pretrained with imagenet. On a first try with few epochs it seems to perform 
better than the model based on trashnet architecture.

The list of layers with corresponding input/output shape is the following: \
```
vgg19
trainable: False
input_shape: (None, 224, 224, 3)
output_shape: (None, 512)
_____________
dense_1
trainable: True
input_shape: (None, 512)
output_shape: (None, 256)
_____________
dropout_1
trainable: True
input_shape: (None, 256)
output_shape: (None, 256)
_____________
batch_normalization_1
trainable: True
input_shape: (None, 256)
output_shape: (None, 256)
_____________
dense_2
trainable: True
input_shape: (None, 256)
output_shape: (None, 6)
_____________
```

Here are the results with 3 epochs without weights regularization:
```
903s 13s/step - loss: 1.6820 - acc: 0.3509 - val_loss: 1.2836 - val_acc: 0.5495
909s 14s/step - loss: 1.3316 - acc: 0.4778 - val_loss: 1.1821 - val_acc: 0.5286
894s 13s/step - loss: 1.2114 - acc: 0.5072 - val_loss: 1.1839 - val_acc: 0.5078
```
Validation accuracy is higher than training set accuracy because on the training set we are doing data augmentation by
shear, zoom, horizontal flip, rotation and shift. \
Notice that while the training accuracy improves, the validation accuracy does not. I think that it suffers from 
overfitting, further studies are needed. \
TODO: 
- increase epochs
- try with leakyReLU activation function
- add regularization
- test different epsilon and learning rate 