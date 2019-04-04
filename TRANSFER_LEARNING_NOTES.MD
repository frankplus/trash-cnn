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
[[44  0  0 15  2  0]
 [ 1 49 10  4 12  0]
 [ 1  8 41  5  7  0]
 [10  1  2 70  7  0]
 [ 2  5  8 21 37  0]
 [ 4  6  1  6  5  0]]
Classification Report
              precision    recall  f1-score   support

   cardboard       0.71      0.72      0.72        61
       glass       0.71      0.64      0.68        76
       metal       0.66      0.66      0.66        62
       paper       0.58      0.78      0.66        90
     plastic       0.53      0.51      0.52        73
       trash       0.00      0.00      0.00        22

   micro avg       0.63      0.63      0.63       384
   macro avg       0.53      0.55      0.54       384
weighted avg       0.60      0.63      0.61       384

UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
```
### Medium period - 50 epochs
- L2 regularization: 0.01
- learning rate = 0.005
- VGG-19 network last layers unfrozen

```
Epoch 1/50
67/67 [==============================] - 506s 8s/step - loss: 2.3966 - acc: 0.3402 - val_loss: 1.8587 - val_acc: 0.3932
Epoch 2/50
67/67 [==============================] - 41s 608ms/step - loss: 1.7978 - acc: 0.3999 - val_loss: 1.8783 - val_acc: 0.3490
Epoch 3/50
67/67 [==============================] - 45s 665ms/step - loss: 1.6417 - acc: 0.4457 - val_loss: 1.5094 - val_acc: 0.4792
Epoch 4/50
67/67 [==============================] - 45s 668ms/step - loss: 1.4916 - acc: 0.5334 - val_loss: 1.7918 - val_acc: 0.3359
Epoch 5/50
67/67 [==============================] - 45s 669ms/step - loss: 1.5090 - acc: 0.5497 - val_loss: 1.6626 - val_acc: 0.4974
Epoch 6/50
67/67 [==============================] - 45s 669ms/step - loss: 1.5220 - acc: 0.5726 - val_loss: 2.0897 - val_acc: 0.4219
Epoch 7/50
67/67 [==============================] - 45s 665ms/step - loss: 1.6098 - acc: 0.4896 - val_loss: 2.6220 - val_acc: 0.2005
Epoch 8/50
67/67 [==============================] - 45s 667ms/step - loss: 1.4697 - acc: 0.5619 - val_loss: 2.9948 - val_acc: 0.2969
Epoch 9/50
67/67 [==============================] - 45s 667ms/step - loss: 1.5537 - acc: 0.5342 - val_loss: 1.6204 - val_acc: 0.5365
Epoch 10/50
67/67 [==============================] - 45s 670ms/step - loss: 1.4848 - acc: 0.5637 - val_loss: 1.5042 - val_acc: 0.5573
Epoch 11/50
67/67 [==============================] - 45s 667ms/step - loss: 1.4630 - acc: 0.5679 - val_loss: 1.5000 - val_acc: 0.5339
Epoch 12/50
67/67 [==============================] - 44s 663ms/step - loss: 1.4012 - acc: 0.5991 - val_loss: 1.5042 - val_acc: 0.5703
Epoch 13/50
67/67 [==============================] - 45s 670ms/step - loss: 1.4968 - acc: 0.5767 - val_loss: 1.6857 - val_acc: 0.5521
Epoch 14/50
67/67 [==============================] - 45s 667ms/step - loss: 1.5000 - acc: 0.5968 - val_loss: 1.5633 - val_acc: 0.5625
Epoch 15/50
67/67 [==============================] - 45s 665ms/step - loss: 1.4328 - acc: 0.6075 - val_loss: 1.3888 - val_acc: 0.5964
Epoch 16/50
67/67 [==============================] - 45s 668ms/step - loss: 1.3258 - acc: 0.6477 - val_loss: 1.5512 - val_acc: 0.5729
Epoch 17/50
67/67 [==============================] - 45s 673ms/step - loss: 1.3022 - acc: 0.6501 - val_loss: 1.3515 - val_acc: 0.6354
Epoch 18/50
67/67 [==============================] - 45s 667ms/step - loss: 1.3127 - acc: 0.6445 - val_loss: 1.3257 - val_acc: 0.6172
Epoch 19/50
67/67 [==============================] - 45s 667ms/step - loss: 1.2415 - acc: 0.6608 - val_loss: 1.2840 - val_acc: 0.6484
Epoch 20/50
67/67 [==============================] - 45s 675ms/step - loss: 1.2577 - acc: 0.6510 - val_loss: 1.5145 - val_acc: 0.5703
Epoch 21/50
67/67 [==============================] - 45s 672ms/step - loss: 1.3553 - acc: 0.6393 - val_loss: 1.4454 - val_acc: 0.6823
Epoch 22/50
67/67 [==============================] - 45s 666ms/step - loss: 1.4186 - acc: 0.6165 - val_loss: 1.4427 - val_acc: 0.6484
Epoch 23/50
67/67 [==============================] - 45s 669ms/step - loss: 1.3386 - acc: 0.6557 - val_loss: 1.4897 - val_acc: 0.6510
Epoch 24/50
67/67 [==============================] - 45s 667ms/step - loss: 1.2996 - acc: 0.6705 - val_loss: 1.5206 - val_acc: 0.5729
Epoch 25/50
67/67 [==============================] - 45s 673ms/step - loss: 1.2933 - acc: 0.6668 - val_loss: 1.6689 - val_acc: 0.5234
Epoch 26/50
67/67 [==============================] - 45s 670ms/step - loss: 1.2356 - acc: 0.6845 - val_loss: 1.6083 - val_acc: 0.5573
Epoch 27/50
67/67 [==============================] - 45s 673ms/step - loss: 1.2054 - acc: 0.6971 - val_loss: 1.5792 - val_acc: 0.5833
Epoch 28/50
67/67 [==============================] - 45s 667ms/step - loss: 1.1806 - acc: 0.7177 - val_loss: 1.3070 - val_acc: 0.7109
Epoch 29/50
67/67 [==============================] - 45s 671ms/step - loss: 1.0653 - acc: 0.7559 - val_loss: 1.2114 - val_acc: 0.7240
Epoch 30/50
67/67 [==============================] - 45s 671ms/step - loss: 1.0318 - acc: 0.7504 - val_loss: 1.2025 - val_acc: 0.7057
Epoch 31/50
67/67 [==============================] - 45s 671ms/step - loss: 0.9830 - acc: 0.7592 - val_loss: 1.2881 - val_acc: 0.6771
Epoch 32/50
67/67 [==============================] - 45s 666ms/step - loss: 0.9480 - acc: 0.7779 - val_loss: 1.2209 - val_acc: 0.7083
Epoch 33/50
67/67 [==============================] - 45s 668ms/step - loss: 0.9272 - acc: 0.7751 - val_loss: 1.1707 - val_acc: 0.7057
Epoch 34/50
67/67 [==============================] - 45s 672ms/step - loss: 0.9707 - acc: 0.7564 - val_loss: 1.3933 - val_acc: 0.6589
Epoch 35/50
67/67 [==============================] - 45s 668ms/step - loss: 0.9680 - acc: 0.7508 - val_loss: 1.2210 - val_acc: 0.6927
Epoch 36/50
67/67 [==============================] - 45s 671ms/step - loss: 0.8929 - acc: 0.7858 - val_loss: 1.1769 - val_acc: 0.7135
Epoch 37/50
67/67 [==============================] - 45s 671ms/step - loss: 0.9060 - acc: 0.7700 - val_loss: 1.2098 - val_acc: 0.7135
Epoch 38/50
67/67 [==============================] - 45s 670ms/step - loss: 0.8493 - acc: 0.7886 - val_loss: 1.1962 - val_acc: 0.7161
Epoch 39/50
67/67 [==============================] - 45s 669ms/step - loss: 0.8619 - acc: 0.7942 - val_loss: 1.2183 - val_acc: 0.7135
Epoch 40/50
67/67 [==============================] - 45s 674ms/step - loss: 0.7735 - acc: 0.8059 - val_loss: 1.1896 - val_acc: 0.7057
Epoch 41/50
67/67 [==============================] - 45s 670ms/step - loss: 0.8064 - acc: 0.8035 - val_loss: 1.1310 - val_acc: 0.7188
Epoch 42/50
67/67 [==============================] - 45s 667ms/step - loss: 0.8013 - acc: 0.7914 - val_loss: 1.3601 - val_acc: 0.7083
Epoch 43/50
67/67 [==============================] - 45s 668ms/step - loss: 0.7986 - acc: 0.8012 - val_loss: 1.2021 - val_acc: 0.7057
Epoch 44/50
67/67 [==============================] - 45s 670ms/step - loss: 0.7529 - acc: 0.8143 - val_loss: 1.1674 - val_acc: 0.7188
Epoch 45/50
67/67 [==============================] - 45s 666ms/step - loss: 0.8163 - acc: 0.7927 - val_loss: 1.3190 - val_acc: 0.7240
Epoch 46/50
67/67 [==============================] - 45s 665ms/step - loss: 0.8228 - acc: 0.7811 - val_loss: 1.5590 - val_acc: 0.6745
Epoch 47/50
67/67 [==============================] - 45s 670ms/step - loss: 0.8926 - acc: 0.7826 - val_loss: 1.3218 - val_acc: 0.7083
Epoch 48/50
67/67 [==============================] - 45s 667ms/step - loss: 0.8226 - acc: 0.8017 - val_loss: 1.4035 - val_acc: 0.6901
Epoch 49/50
67/67 [==============================] - 45s 667ms/step - loss: 0.7970 - acc: 0.8119 - val_loss: 1.2295 - val_acc: 0.7318
Epoch 50/50
67/67 [==============================] - 45s 668ms/step - loss: 0.8599 - acc: 0.7868 - val_loss: 1.3382 - val_acc: 0.7005

Confusion Matrix
[[46  1  0 14  0  0]
 [ 0 48  9  3 16  0]
 [ 0  4 51  3  2  2]
 [ 2  0  2 79  3  4]
 [ 0  4  5 16 41  7]
 [ 2  3  3  8  2  4]]
Classification Report
              precision    recall  f1-score   support

   cardboard       0.92      0.75      0.83        61
       glass       0.80      0.63      0.71        76
       metal       0.73      0.82      0.77        62
       paper       0.64      0.88      0.74        90
     plastic       0.64      0.56      0.60        73
       trash       0.24      0.18      0.21        22

   micro avg       0.70      0.70      0.70       384
   macro avg       0.66      0.64      0.64       384
weighted avg       0.71      0.70      0.70       384

```
