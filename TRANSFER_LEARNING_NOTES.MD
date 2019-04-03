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

### First training period - 10 epochs: 
- no regularization
- learning rate = 0.001
- VGG-19 network frozen (trainable = false)
```
Epoch 1/10
67/67 [==============================] - 309s 5s/step - loss: 1.6383 - acc: 0.3729 - val_loss: 1.2918 - val_acc: 0.5026
Epoch 2/10
67/67 [==============================] - 40s 594ms/step - loss: 1.2966 - acc: 0.4722 - val_loss: 1.2453 - val_acc: 0.4896
Epoch 3/10
67/67 [==============================] - 43s 644ms/step - loss: 1.2107 - acc: 0.5221 - val_loss: 1.1678 - val_acc: 0.5495
Epoch 4/10
67/67 [==============================] - 43s 646ms/step - loss: 1.1640 - acc: 0.5426 - val_loss: 1.2801 - val_acc: 0.5156
Epoch 5/10
67/67 [==============================] - 43s 644ms/step - loss: 1.1487 - acc: 0.5479 - val_loss: 1.1942 - val_acc: 0.5391
Epoch 6/10
67/67 [==============================] - 44s 654ms/step - loss: 1.1394 - acc: 0.5558 - val_loss: 1.1373 - val_acc: 0.5781
Epoch 7/10
67/67 [==============================] - 43s 644ms/step - loss: 1.1168 - acc: 0.5730 - val_loss: 1.1129 - val_acc: 0.5755
Epoch 8/10
67/67 [==============================] - 43s 644ms/step - loss: 1.1283 - acc: 0.5525 - val_loss: 1.1070 - val_acc: 0.5964
Epoch 9/10
67/67 [==============================] - 43s 645ms/step - loss: 1.1159 - acc: 0.5618 - val_loss: 1.1212 - val_acc: 0.6094
Epoch 10/10
67/67 [==============================] - 43s 644ms/step - loss: 1.1070 - acc: 0.5726 - val_loss: 1.0915 - val_acc: 0.6276

Confusion Matrix
[[ 0  0 28  0 13 20]
 [ 0  2 18  0 11 45]
 [ 0 11 12  0 10 29]
 [ 0  2 38  0 14 36]
 [ 0  8 25  0  5 35]
 [ 0  1 14  0  1  6]]
Classification Report
              precision    recall  f1-score   support

   cardboard       0.00      0.00      0.00        61
       glass       0.08      0.03      0.04        76
       metal       0.09      0.19      0.12        62
       paper       0.00      0.00      0.00        90
     plastic       0.09      0.07      0.08        73
       trash       0.04      0.27      0.06        22

   micro avg       0.07      0.07      0.07       384
   macro avg       0.05      0.09      0.05       384
weighted avg       0.05      0.07      0.05       384

UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
```
### Medium period - 50 epochs
- L2 regularization: 0.01
- learning rate = 0.005
- VGG-19 network last layers unfrozen

.......... I AM COMPUTING ............
