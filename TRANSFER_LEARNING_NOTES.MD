## Using transfer learning
Following [this paper](http://cs230.stanford.edu/projects_spring_2018/reports/8290808.pdf) we tried to use transfer 
learning using the VGG-19 architecture pretrained with imagenet. On a first try with few epochs it seems to perform 
better than the model based on trashnet architecture.

The list of layers with corresponding input/output shape is the following:
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
### Medium training period - 50 epochs
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

### First long training period - 50 epochs
- L2 regularization: 0.05
- learning rate = 0.0001
- VGG-19 network last layers unfrozen

```
Epoch 1/50
67/67 [==============================] - 618s 9s/step - loss: 1.7898 - acc: 0.8297 - val_loss: 2.2494 - val_acc: 0.7188
Epoch 2/50
67/67 [==============================] - 42s 627ms/step - loss: 1.6593 - acc: 0.8250 - val_loss: 2.1326 - val_acc: 0.7214
Epoch 3/50
67/67 [==============================] - 46s 683ms/step - loss: 1.5743 - acc: 0.8227 - val_loss: 2.0404 - val_acc: 0.7214
Epoch 4/50
67/67 [==============================] - 46s 687ms/step - loss: 1.4771 - acc: 0.8255 - val_loss: 1.9495 - val_acc: 0.7344
Epoch 5/50
67/67 [==============================] - 46s 683ms/step - loss: 1.3918 - acc: 0.8232 - val_loss: 1.8869 - val_acc: 0.7344
Epoch 6/50
67/67 [==============================] - 45s 677ms/step - loss: 1.2910 - acc: 0.8348 - val_loss: 1.8179 - val_acc: 0.7292
Epoch 7/50
67/67 [==============================] - 46s 683ms/step - loss: 1.2579 - acc: 0.8311 - val_loss: 1.7598 - val_acc: 0.7344
Epoch 8/50
67/67 [==============================] - 46s 682ms/step - loss: 1.1778 - acc: 0.8428 - val_loss: 1.7048 - val_acc: 0.7370
Epoch 9/50
67/67 [==============================] - 46s 685ms/step - loss: 1.1359 - acc: 0.8357 - val_loss: 1.6636 - val_acc: 0.7318
Epoch 10/50
67/67 [==============================] - 45s 679ms/step - loss: 1.0751 - acc: 0.8460 - val_loss: 1.6207 - val_acc: 0.7344
Epoch 11/50
67/67 [==============================] - 45s 674ms/step - loss: 1.0426 - acc: 0.8347 - val_loss: 1.5841 - val_acc: 0.7370
Epoch 12/50
67/67 [==============================] - 45s 679ms/step - loss: 0.9983 - acc: 0.8441 - val_loss: 1.5559 - val_acc: 0.7292
Epoch 13/50
67/67 [==============================] - 45s 672ms/step - loss: 0.9659 - acc: 0.8507 - val_loss: 1.5155 - val_acc: 0.7292
Epoch 14/50
67/67 [==============================] - 45s 673ms/step - loss: 0.9316 - acc: 0.8465 - val_loss: 1.4847 - val_acc: 0.7292
Epoch 15/50
67/67 [==============================] - 45s 674ms/step - loss: 0.9025 - acc: 0.8516 - val_loss: 1.4626 - val_acc: 0.7318
Epoch 16/50
67/67 [==============================] - 45s 673ms/step - loss: 0.8967 - acc: 0.8488 - val_loss: 1.4472 - val_acc: 0.7292
Epoch 17/50
67/67 [==============================] - 46s 680ms/step - loss: 0.8578 - acc: 0.8493 - val_loss: 1.4259 - val_acc: 0.7370
Epoch 18/50
67/67 [==============================] - 45s 678ms/step - loss: 0.8110 - acc: 0.8535 - val_loss: 1.4000 - val_acc: 0.7266
Epoch 19/50
67/67 [==============================] - 46s 689ms/step - loss: 0.8041 - acc: 0.8539 - val_loss: 1.3758 - val_acc: 0.7292
Epoch 20/50
67/67 [==============================] - 46s 690ms/step - loss: 0.7778 - acc: 0.8605 - val_loss: 1.3838 - val_acc: 0.7266
Epoch 21/50
67/67 [==============================] - 46s 683ms/step - loss: 0.7652 - acc: 0.8582 - val_loss: 1.3607 - val_acc: 0.7292
Epoch 22/50
67/67 [==============================] - 46s 684ms/step - loss: 0.7320 - acc: 0.8540 - val_loss: 1.3500 - val_acc: 0.7344
Epoch 23/50
67/67 [==============================] - 46s 685ms/step - loss: 0.7255 - acc: 0.8591 - val_loss: 1.3398 - val_acc: 0.7344
Epoch 24/50
67/67 [==============================] - 46s 691ms/step - loss: 0.7194 - acc: 0.8623 - val_loss: 1.3240 - val_acc: 0.7318
Epoch 25/50
67/67 [==============================] - 46s 688ms/step - loss: 0.6934 - acc: 0.8633 - val_loss: 1.3231 - val_acc: 0.7344
Epoch 26/50
67/67 [==============================] - 46s 683ms/step - loss: 0.6657 - acc: 0.8698 - val_loss: 1.3397 - val_acc: 0.7292
Epoch 27/50
67/67 [==============================] - 45s 676ms/step - loss: 0.6469 - acc: 0.8721 - val_loss: 1.3197 - val_acc: 0.7318
Epoch 28/50
67/67 [==============================] - 45s 679ms/step - loss: 0.6598 - acc: 0.8618 - val_loss: 1.2876 - val_acc: 0.7318
Epoch 29/50
67/67 [==============================] - 45s 676ms/step - loss: 0.6326 - acc: 0.8749 - val_loss: 1.2786 - val_acc: 0.7370
Epoch 30/50
67/67 [==============================] - 46s 681ms/step - loss: 0.6365 - acc: 0.8703 - val_loss: 1.3080 - val_acc: 0.7396
Epoch 31/50
67/67 [==============================] - 45s 673ms/step - loss: 0.6368 - acc: 0.8623 - val_loss: 1.2875 - val_acc: 0.7396
Epoch 32/50
67/67 [==============================] - 45s 676ms/step - loss: 0.6263 - acc: 0.8619 - val_loss: 1.2717 - val_acc: 0.7370
Epoch 33/50
67/67 [==============================] - 46s 690ms/step - loss: 0.5974 - acc: 0.8736 - val_loss: 1.2641 - val_acc: 0.7422
Epoch 34/50
67/67 [==============================] - 46s 683ms/step - loss: 0.5931 - acc: 0.8745 - val_loss: 1.2574 - val_acc: 0.7448
Epoch 35/50
67/67 [==============================] - 46s 680ms/step - loss: 0.5666 - acc: 0.8755 - val_loss: 1.2583 - val_acc: 0.7370
Epoch 36/50
67/67 [==============================] - 45s 676ms/step - loss: 0.5720 - acc: 0.8814 - val_loss: 1.2481 - val_acc: 0.7396
Epoch 37/50
67/67 [==============================] - 45s 677ms/step - loss: 0.5793 - acc: 0.8670 - val_loss: 1.2580 - val_acc: 0.7396
Epoch 38/50
67/67 [==============================] - 45s 673ms/step - loss: 0.5521 - acc: 0.8740 - val_loss: 1.2345 - val_acc: 0.7422
Epoch 39/50
67/67 [==============================] - 45s 674ms/step - loss: 0.5299 - acc: 0.8820 - val_loss: 1.2381 - val_acc: 0.7292
Epoch 40/50
67/67 [==============================] - 45s 679ms/step - loss: 0.5039 - acc: 0.8908 - val_loss: 1.2353 - val_acc: 0.7344
Epoch 41/50
67/67 [==============================] - 46s 683ms/step - loss: 0.5246 - acc: 0.8801 - val_loss: 1.2316 - val_acc: 0.7370
Epoch 42/50
67/67 [==============================] - 46s 685ms/step - loss: 0.5177 - acc: 0.8745 - val_loss: 1.2286 - val_acc: 0.7370
Epoch 43/50
67/67 [==============================] - 46s 681ms/step - loss: 0.5183 - acc: 0.8825 - val_loss: 1.2130 - val_acc: 0.7344
Epoch 44/50
67/67 [==============================] - 45s 678ms/step - loss: 0.5195 - acc: 0.8735 - val_loss: 1.2234 - val_acc: 0.7422
Epoch 45/50
67/67 [==============================] - 45s 676ms/step - loss: 0.5170 - acc: 0.8768 - val_loss: 1.2123 - val_acc: 0.7422
Epoch 46/50
67/67 [==============================] - 45s 670ms/step - loss: 0.4989 - acc: 0.8754 - val_loss: 1.2151 - val_acc: 0.7344
Epoch 47/50
67/67 [==============================] - 46s 681ms/step - loss: 0.5115 - acc: 0.8810 - val_loss: 1.2120 - val_acc: 0.7344
Epoch 48/50
67/67 [==============================] - 46s 681ms/step - loss: 0.5018 - acc: 0.8829 - val_loss: 1.2070 - val_acc: 0.7396
Epoch 49/50
67/67 [==============================] - 45s 678ms/step - loss: 0.4907 - acc: 0.8829 - val_loss: 1.2044 - val_acc: 0.7370
Epoch 50/50
67/67 [==============================] - 46s 681ms/step - loss: 0.4669 - acc: 0.8922 - val_loss: 1.2058 - val_acc: 0.7422
```

### Second long training period - 50 epochs
- L2 regularization: 0.1
- learning rate = 0.0001
- VGG-19 network last layers unfrozen

```
Epoch 1/50
67/67 [==============================] - 48s 723ms/step - loss: 0.5840 - acc: 0.8857 - val_loss: 1.3369 - val_acc: 0.7396
Epoch 2/50
67/67 [==============================] - 45s 675ms/step - loss: 0.5761 - acc: 0.8918 - val_loss: 1.3209 - val_acc: 0.7396
Epoch 3/50
67/67 [==============================] - 45s 676ms/step - loss: 0.5587 - acc: 0.8824 - val_loss: 1.3226 - val_acc: 0.7396
Epoch 4/50
67/67 [==============================] - 46s 680ms/step - loss: 0.5565 - acc: 0.8842 - val_loss: 1.2929 - val_acc: 0.7370
Epoch 5/50
67/67 [==============================] - 45s 673ms/step - loss: 0.5604 - acc: 0.8875 - val_loss: 1.2756 - val_acc: 0.7448
Epoch 6/50
67/67 [==============================] - 45s 677ms/step - loss: 0.5490 - acc: 0.8833 - val_loss: 1.2903 - val_acc: 0.7318
Epoch 7/50
67/67 [==============================] - 46s 682ms/step - loss: 0.5174 - acc: 0.8950 - val_loss: 1.2859 - val_acc: 0.7396
Epoch 8/50
67/67 [==============================] - 45s 673ms/step - loss: 0.5022 - acc: 0.8950 - val_loss: 1.2785 - val_acc: 0.7396
Epoch 9/50
67/67 [==============================] - 45s 668ms/step - loss: 0.5318 - acc: 0.8786 - val_loss: 1.2581 - val_acc: 0.7396
Epoch 10/50
67/67 [==============================] - 45s 669ms/step - loss: 0.5200 - acc: 0.8834 - val_loss: 1.2545 - val_acc: 0.7396
Epoch 11/50
67/67 [==============================] - 45s 673ms/step - loss: 0.5020 - acc: 0.8852 - val_loss: 1.2504 - val_acc: 0.7370
Epoch 12/50
67/67 [==============================] - 45s 675ms/step - loss: 0.5275 - acc: 0.8800 - val_loss: 1.2367 - val_acc: 0.7396
Epoch 13/50
67/67 [==============================] - 45s 672ms/step - loss: 0.4867 - acc: 0.8838 - val_loss: 1.2470 - val_acc: 0.7396
Epoch 14/50
67/67 [==============================] - 45s 669ms/step - loss: 0.4812 - acc: 0.8941 - val_loss: 1.2316 - val_acc: 0.7396
Epoch 15/50
67/67 [==============================] - 45s 673ms/step - loss: 0.4802 - acc: 0.8973 - val_loss: 1.2255 - val_acc: 0.7422
Epoch 16/50
67/67 [==============================] - 45s 672ms/step - loss: 0.4818 - acc: 0.8867 - val_loss: 1.2199 - val_acc: 0.7448
Epoch 17/50
67/67 [==============================] - 45s 674ms/step - loss: 0.4707 - acc: 0.8931 - val_loss: 1.2183 - val_acc: 0.7422
Epoch 18/50
67/67 [==============================] - 45s 672ms/step - loss: 0.4700 - acc: 0.9020 - val_loss: 1.2303 - val_acc: 0.7474
Epoch 19/50
67/67 [==============================] - 45s 666ms/step - loss: 0.4517 - acc: 0.8927 - val_loss: 1.2097 - val_acc: 0.7448
Epoch 20/50
67/67 [==============================] - 45s 668ms/step - loss: 0.4476 - acc: 0.8926 - val_loss: 1.2285 - val_acc: 0.7448
Epoch 21/50
67/67 [==============================] - 45s 671ms/step - loss: 0.4440 - acc: 0.8917 - val_loss: 1.2250 - val_acc: 0.7474
Epoch 22/50
67/67 [==============================] - 45s 671ms/step - loss: 0.4275 - acc: 0.8908 - val_loss: 1.2159 - val_acc: 0.7500
Epoch 23/50
67/67 [==============================] - 45s 671ms/step - loss: 0.4247 - acc: 0.8936 - val_loss: 1.2164 - val_acc: 0.7500
Epoch 24/50
67/67 [==============================] - 45s 677ms/step - loss: 0.4277 - acc: 0.8935 - val_loss: 1.2203 - val_acc: 0.7526
Epoch 25/50
67/67 [==============================] - 45s 679ms/step - loss: 0.4130 - acc: 0.8964 - val_loss: 1.2047 - val_acc: 0.7474
Epoch 26/50
67/67 [==============================] - 45s 672ms/step - loss: 0.4097 - acc: 0.9005 - val_loss: 1.2111 - val_acc: 0.7448
Epoch 27/50
67/67 [==============================] - 45s 679ms/step - loss: 0.4200 - acc: 0.8964 - val_loss: 1.2038 - val_acc: 0.7526
Epoch 28/50
67/67 [==============================] - 46s 679ms/step - loss: 0.4314 - acc: 0.8945 - val_loss: 1.2052 - val_acc: 0.7500
Epoch 29/50
67/67 [==============================] - 46s 680ms/step - loss: 0.4236 - acc: 0.8969 - val_loss: 1.2136 - val_acc: 0.7448
Epoch 30/50
67/67 [==============================] - 46s 686ms/step - loss: 0.4054 - acc: 0.9034 - val_loss: 1.1965 - val_acc: 0.7526
Epoch 31/50
67/67 [==============================] - 46s 685ms/step - loss: 0.3969 - acc: 0.9100 - val_loss: 1.2123 - val_acc: 0.7526
Epoch 32/50
67/67 [==============================] - 46s 684ms/step - loss: 0.3977 - acc: 0.9002 - val_loss: 1.2031 - val_acc: 0.7552
Epoch 33/50
67/67 [==============================] - 46s 684ms/step - loss: 0.3958 - acc: 0.8983 - val_loss: 1.1914 - val_acc: 0.7500
Epoch 34/50
67/67 [==============================] - 46s 688ms/step - loss: 0.4206 - acc: 0.8917 - val_loss: 1.1884 - val_acc: 0.7500
Epoch 35/50
67/67 [==============================] - 46s 689ms/step - loss: 0.4020 - acc: 0.8978 - val_loss: 1.1965 - val_acc: 0.7448
Epoch 36/50
67/67 [==============================] - 46s 688ms/step - loss: 0.3835 - acc: 0.9002 - val_loss: 1.1850 - val_acc: 0.7448
Epoch 37/50
67/67 [==============================] - 46s 688ms/step - loss: 0.3724 - acc: 0.9090 - val_loss: 1.1656 - val_acc: 0.7474
Epoch 38/50
67/67 [==============================] - 46s 685ms/step - loss: 0.3859 - acc: 0.9062 - val_loss: 1.1720 - val_acc: 0.7500
Epoch 39/50
67/67 [==============================] - 46s 682ms/step - loss: 0.4108 - acc: 0.8983 - val_loss: 1.1828 - val_acc: 0.7422
Epoch 40/50
67/67 [==============================] - 46s 685ms/step - loss: 0.3948 - acc: 0.8964 - val_loss: 1.1710 - val_acc: 0.7474
Epoch 41/50
67/67 [==============================] - 46s 686ms/step - loss: 0.3929 - acc: 0.8950 - val_loss: 1.2008 - val_acc: 0.7422
Epoch 42/50
67/67 [==============================] - 46s 683ms/step - loss: 0.4023 - acc: 0.8936 - val_loss: 1.1888 - val_acc: 0.7552
Epoch 43/50
67/67 [==============================] - 46s 679ms/step - loss: 0.3802 - acc: 0.8997 - val_loss: 1.2095 - val_acc: 0.7422
Epoch 44/50
67/67 [==============================] - 46s 687ms/step - loss: 0.3798 - acc: 0.8988 - val_loss: 1.1913 - val_acc: 0.7448
Epoch 45/50
67/67 [==============================] - 46s 688ms/step - loss: 0.3831 - acc: 0.8941 - val_loss: 1.1873 - val_acc: 0.7422
Epoch 46/50
67/67 [==============================] - 46s 689ms/step - loss: 0.3569 - acc: 0.9076 - val_loss: 1.1720 - val_acc: 0.7396
Epoch 47/50
67/67 [==============================] - 46s 680ms/step - loss: 0.3892 - acc: 0.8973 - val_loss: 1.1510 - val_acc: 0.7422
Epoch 48/50
67/67 [==============================] - 46s 680ms/step - loss: 0.3618 - acc: 0.8969 - val_loss: 1.1601 - val_acc: 0.7474
Epoch 49/50
67/67 [==============================] - 45s 676ms/step - loss: 0.3322 - acc: 0.9109 - val_loss: 1.1817 - val_acc: 0.7500
Epoch 50/50
67/67 [==============================] - 46s 688ms/step - loss: 0.3610 - acc: 0.9066 - val_loss: 1.1601 - val_acc: 0.7526

Confusion Matrix
[[56  0  0  5  0  0]
 [ 0 61  2  0 13  0]
 [ 1  7 48  0  3  3]
 [ 5  2  1 73  4  5]
 [ 0  2  3 11 46 11]
 [ 3  4  1  7  2  5]]
Classification Report
              precision    recall  f1-score   support

   cardboard       0.86      0.92      0.89        61
       glass       0.80      0.80      0.80        76
       metal       0.87      0.77      0.82        62
       paper       0.76      0.81      0.78        90
     plastic       0.68      0.63      0.65        73
       trash       0.21      0.23      0.22        22

   micro avg       0.75      0.75      0.75       384
   macro avg       0.70      0.69      0.69       384
weighted avg       0.76      0.75      0.75       384

```

