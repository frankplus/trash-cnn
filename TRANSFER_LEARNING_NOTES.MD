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
- L2 weights regularization: 0.001
- learning rate = 0.001
- VGG-19 network last layers unfrozen
- epsilon = 0.5

```
Epoch 1/10
67/67 [==============================] - 50s 741ms/step - loss: 2.6628 - acc: 0.1591 - val_loss: 2.2411 - val_acc: 0.1849
Epoch 2/10
67/67 [==============================] - 44s 661ms/step - loss: 2.4977 - acc: 0.2081 - val_loss: 2.0965 - val_acc: 0.2630
Epoch 3/10
67/67 [==============================] - 44s 657ms/step - loss: 2.4109 - acc: 0.2310 - val_loss: 1.9504 - val_acc: 0.3568
Epoch 4/10
67/67 [==============================] - 45s 671ms/step - loss: 2.2184 - acc: 0.2655 - val_loss: 1.9077 - val_acc: 0.4062
Epoch 5/10
67/67 [==============================] - 45s 669ms/step - loss: 2.1386 - acc: 0.3154 - val_loss: 1.8048 - val_acc: 0.4479
Epoch 6/10
67/67 [==============================] - 44s 660ms/step - loss: 2.0609 - acc: 0.3421 - val_loss: 1.7134 - val_acc: 0.5260
Epoch 7/10
67/67 [==============================] - 44s 657ms/step - loss: 1.9832 - acc: 0.3612 - val_loss: 1.6683 - val_acc: 0.5208
Epoch 8/10
67/67 [==============================] - 45s 666ms/step - loss: 1.8577 - acc: 0.4065 - val_loss: 1.6237 - val_acc: 0.5573
Epoch 9/10
67/67 [==============================] - 45s 667ms/step - loss: 1.8622 - acc: 0.4191 - val_loss: 1.5994 - val_acc: 0.5547
Epoch 10/10
67/67 [==============================] - 45s 669ms/step - loss: 1.7948 - acc: 0.4223 - val_loss: 1.5557 - val_acc: 0.5729
Confusion Matrix
[[44  1  1 13  2  0]
 [ 2 37 24  2 10  1]
 [ 1  3 50  5  2  1]
 [12  2  5 56 12  3]
 [ 1  6 19 12 33  2]
 [ 2  5  4  2  9  0]]
Classification Report
              precision    recall  f1-score   support

   cardboard       0.71      0.72      0.72        61
       glass       0.69      0.49      0.57        76
       metal       0.49      0.81      0.61        62
       paper       0.62      0.62      0.62        90
     plastic       0.49      0.45      0.47        73
       trash       0.00      0.00      0.00        22

   micro avg       0.57      0.57      0.57       384
   macro avg       0.50      0.51      0.50       384
weighted avg       0.56      0.57      0.56       384
```

### Medium training period - 50 epochs
- L2 weights regularization: 0.01
- learning rate = 0.001
- VGG-19 network last layers unfrozen
- epsilon = 0.5

```
Epoch 1/50
67/67 [==============================] - 51s 757ms/step - loss: 4.8202 - acc: 0.4386 - val_loss: 4.6074 - val_acc: 0.5911
Epoch 2/50
67/67 [==============================] - 46s 685ms/step - loss: 4.8242 - acc: 0.4489 - val_loss: 4.6020 - val_acc: 0.5781
Epoch 3/50
67/67 [==============================] - 46s 687ms/step - loss: 4.7559 - acc: 0.4685 - val_loss: 4.5720 - val_acc: 0.5859
Epoch 4/50
67/67 [==============================] - 47s 694ms/step - loss: 4.7239 - acc: 0.4863 - val_loss: 4.5566 - val_acc: 0.5885
Epoch 5/50
67/67 [==============================] - 46s 693ms/step - loss: 4.6762 - acc: 0.4839 - val_loss: 4.5301 - val_acc: 0.6016
Epoch 6/50
67/67 [==============================] - 46s 683ms/step - loss: 4.6566 - acc: 0.4952 - val_loss: 4.4971 - val_acc: 0.5990
Epoch 7/50
67/67 [==============================] - 45s 678ms/step - loss: 4.6223 - acc: 0.5119 - val_loss: 4.4834 - val_acc: 0.5938
Epoch 8/50
67/67 [==============================] - 45s 676ms/step - loss: 4.5780 - acc: 0.5259 - val_loss: 4.4726 - val_acc: 0.6042
Epoch 9/50
67/67 [==============================] - 45s 678ms/step - loss: 4.5752 - acc: 0.5231 - val_loss: 4.4703 - val_acc: 0.5938
Epoch 10/50
67/67 [==============================] - 45s 678ms/step - loss: 4.5365 - acc: 0.5348 - val_loss: 4.4409 - val_acc: 0.6094
Epoch 11/50
67/67 [==============================] - 45s 678ms/step - loss: 4.5190 - acc: 0.5240 - val_loss: 4.3923 - val_acc: 0.6328
Epoch 12/50
67/67 [==============================] - 46s 679ms/step - loss: 4.4312 - acc: 0.5749 - val_loss: 4.3515 - val_acc: 0.6302
Epoch 13/50
67/67 [==============================] - 46s 687ms/step - loss: 4.4169 - acc: 0.5712 - val_loss: 4.3365 - val_acc: 0.6536
Epoch 14/50
67/67 [==============================] - 46s 686ms/step - loss: 4.3818 - acc: 0.5745 - val_loss: 4.3118 - val_acc: 0.6510
Epoch 15/50
67/67 [==============================] - 46s 684ms/step - loss: 4.3470 - acc: 0.5921 - val_loss: 4.3109 - val_acc: 0.6615
Epoch 16/50
67/67 [==============================] - 44s 663ms/step - loss: 4.3294 - acc: 0.5945 - val_loss: 4.2787 - val_acc: 0.6432
Epoch 17/50
67/67 [==============================] - 43s 649ms/step - loss: 4.3213 - acc: 0.5978 - val_loss: 4.2536 - val_acc: 0.6406
Epoch 18/50
67/67 [==============================] - 43s 647ms/step - loss: 4.2767 - acc: 0.6019 - val_loss: 4.2280 - val_acc: 0.6693
Epoch 19/50
67/67 [==============================] - 43s 643ms/step - loss: 4.2383 - acc: 0.6072 - val_loss: 4.2160 - val_acc: 0.6641
Epoch 20/50
67/67 [==============================] - 43s 646ms/step - loss: 4.2141 - acc: 0.6225 - val_loss: 4.2126 - val_acc: 0.6354
Epoch 21/50
67/67 [==============================] - 43s 643ms/step - loss: 4.1744 - acc: 0.6201 - val_loss: 4.1771 - val_acc: 0.6432
Epoch 22/50
67/67 [==============================] - 44s 659ms/step - loss: 4.1315 - acc: 0.6454 - val_loss: 4.1448 - val_acc: 0.6771
Epoch 23/50
67/67 [==============================] - 44s 653ms/step - loss: 4.0994 - acc: 0.6467 - val_loss: 4.1258 - val_acc: 0.6797
Epoch 24/50
67/67 [==============================] - 44s 654ms/step - loss: 4.1083 - acc: 0.6192 - val_loss: 4.0912 - val_acc: 0.6771
Epoch 25/50
67/67 [==============================] - 43s 643ms/step - loss: 4.0452 - acc: 0.6522 - val_loss: 4.0813 - val_acc: 0.6771
Epoch 26/50
67/67 [==============================] - 44s 655ms/step - loss: 4.0377 - acc: 0.6551 - val_loss: 4.0386 - val_acc: 0.6927
Epoch 27/50
67/67 [==============================] - 44s 653ms/step - loss: 4.0142 - acc: 0.6528 - val_loss: 4.0284 - val_acc: 0.7031
Epoch 28/50
67/67 [==============================] - 44s 651ms/step - loss: 3.9658 - acc: 0.6780 - val_loss: 4.0050 - val_acc: 0.7135
Epoch 29/50
67/67 [==============================] - 43s 649ms/step - loss: 3.9431 - acc: 0.6742 - val_loss: 3.9744 - val_acc: 0.6797
Epoch 30/50
67/67 [==============================] - 44s 661ms/step - loss: 3.9323 - acc: 0.6724 - val_loss: 3.9711 - val_acc: 0.6771
Epoch 31/50
67/67 [==============================] - 45s 672ms/step - loss: 3.8897 - acc: 0.6766 - val_loss: 3.9690 - val_acc: 0.6979
Epoch 32/50
67/67 [==============================] - 45s 667ms/step - loss: 3.8850 - acc: 0.6780 - val_loss: 3.9314 - val_acc: 0.6823
Epoch 33/50
67/67 [==============================] - 44s 662ms/step - loss: 3.8327 - acc: 0.7103 - val_loss: 3.9428 - val_acc: 0.6927
Epoch 34/50
67/67 [==============================] - 44s 659ms/step - loss: 3.7922 - acc: 0.7149 - val_loss: 3.9013 - val_acc: 0.7005
Epoch 35/50
67/67 [==============================] - 45s 666ms/step - loss: 3.7682 - acc: 0.7116 - val_loss: 3.9083 - val_acc: 0.6927
Epoch 36/50
67/67 [==============================] - 44s 664ms/step - loss: 3.7613 - acc: 0.7037 - val_loss: 3.8855 - val_acc: 0.7057
Epoch 37/50
67/67 [==============================] - 44s 655ms/step - loss: 3.7646 - acc: 0.7019 - val_loss: 3.8819 - val_acc: 0.6927
Epoch 38/50
67/67 [==============================] - 45s 670ms/step - loss: 3.7348 - acc: 0.7140 - val_loss: 3.8739 - val_acc: 0.6979
Epoch 39/50
67/67 [==============================] - 44s 658ms/step - loss: 3.6854 - acc: 0.7205 - val_loss: 3.8273 - val_acc: 0.6927
Epoch 40/50
67/67 [==============================] - 45s 671ms/step - loss: 3.6559 - acc: 0.7228 - val_loss: 3.7790 - val_acc: 0.7161
Epoch 41/50
67/67 [==============================] - 45s 673ms/step - loss: 3.6323 - acc: 0.7261 - val_loss: 3.7734 - val_acc: 0.7083
Epoch 42/50
67/67 [==============================] - 44s 664ms/step - loss: 3.6583 - acc: 0.7140 - val_loss: 3.7813 - val_acc: 0.7031
Epoch 43/50
67/67 [==============================] - 44s 660ms/step - loss: 3.6077 - acc: 0.7270 - val_loss: 3.7391 - val_acc: 0.7057
Epoch 44/50
67/67 [==============================] - 44s 660ms/step - loss: 3.6031 - acc: 0.7167 - val_loss: 3.7285 - val_acc: 0.6953
Epoch 45/50
67/67 [==============================] - 44s 657ms/step - loss: 3.5765 - acc: 0.7242 - val_loss: 3.7049 - val_acc: 0.7031
Epoch 46/50
67/67 [==============================] - 44s 661ms/step - loss: 3.5632 - acc: 0.7317 - val_loss: 3.7224 - val_acc: 0.7005
Epoch 47/50
67/67 [==============================] - 45s 664ms/step - loss: 3.5098 - acc: 0.7443 - val_loss: 3.6868 - val_acc: 0.7240
Epoch 48/50
67/67 [==============================] - 44s 662ms/step - loss: 3.4809 - acc: 0.7485 - val_loss: 3.6599 - val_acc: 0.7161
Epoch 49/50
67/67 [==============================] - 44s 662ms/step - loss: 3.4956 - acc: 0.7298 - val_loss: 3.6261 - val_acc: 0.7083
Epoch 50/50
67/67 [==============================] - 45s 664ms/step - loss: 3.4991 - acc: 0.7340 - val_loss: 3.6362 - val_acc: 0.7240
Confusion Matrix
[[50  0  1  9  1  0]
 [ 0 54 13  1  8  0]
 [ 0  4 54  1  2  1]
 [ 6  2  1 75  4  2]
 [ 2  4  8 11 42  6]
 [ 0  3  2  7  7  3]]
Classification Report
              precision    recall  f1-score   support

   cardboard       0.86      0.82      0.84        61
       glass       0.81      0.71      0.76        76
       metal       0.68      0.87      0.77        62
       paper       0.72      0.83      0.77        90
     plastic       0.66      0.58      0.61        73
       trash       0.25      0.14      0.18        22

   micro avg       0.72      0.72      0.72       384
   macro avg       0.66      0.66      0.65       384
weighted avg       0.71      0.72      0.71       384
```

### First long training period - 50 epochs
- L2 regularization: 0.05
- learning rate = 0.001
- VGG-19 network last layers unfrozen
- epsilon = 0.5

```
Epoch 1/50
67/67 [==============================] - 50s 743ms/step - loss: 14.4078 - acc: 0.7420 - val_loss: 14.5638 - val_acc: 0.7292
Epoch 2/50
67/67 [==============================] - 45s 665ms/step - loss: 14.3370 - acc: 0.7480 - val_loss: 14.4578 - val_acc: 0.7135
Epoch 3/50
67/67 [==============================] - 45s 670ms/step - loss: 14.2022 - acc: 0.7546 - val_loss: 14.3120 - val_acc: 0.7240
Epoch 4/50
67/67 [==============================] - 45s 671ms/step - loss: 14.0475 - acc: 0.7475 - val_loss: 14.1684 - val_acc: 0.7135
Epoch 5/50
67/67 [==============================] - 45s 668ms/step - loss: 13.8664 - acc: 0.7522 - val_loss: 13.9767 - val_acc: 0.7188
Epoch 6/50
67/67 [==============================] - 45s 671ms/step - loss: 13.6729 - acc: 0.7518 - val_loss: 13.7888 - val_acc: 0.7005
Epoch 7/50
67/67 [==============================] - 45s 672ms/step - loss: 13.4873 - acc: 0.7564 - val_loss: 13.6094 - val_acc: 0.6979
Epoch 8/50
67/67 [==============================] - 45s 672ms/step - loss: 13.2836 - acc: 0.7666 - val_loss: 13.3962 - val_acc: 0.7188
Epoch 9/50
67/67 [==============================] - 45s 668ms/step - loss: 13.0571 - acc: 0.7587 - val_loss: 13.1728 - val_acc: 0.7240
Epoch 10/50
67/67 [==============================] - 45s 678ms/step - loss: 12.8406 - acc: 0.7583 - val_loss: 12.9428 - val_acc: 0.7083
Epoch 11/50
67/67 [==============================] - 46s 682ms/step - loss: 12.5854 - acc: 0.7746 - val_loss: 12.6990 - val_acc: 0.7083
Epoch 12/50
67/67 [==============================] - 45s 669ms/step - loss: 12.3793 - acc: 0.7579 - val_loss: 12.4827 - val_acc: 0.7214
Epoch 13/50
67/67 [==============================] - 45s 670ms/step - loss: 12.1536 - acc: 0.7620 - val_loss: 12.2583 - val_acc: 0.7240
Epoch 14/50
67/67 [==============================] - 44s 664ms/step - loss: 11.9170 - acc: 0.7639 - val_loss: 12.0356 - val_acc: 0.7135
Epoch 15/50
67/67 [==============================] - 44s 662ms/step - loss: 11.6616 - acc: 0.7821 - val_loss: 11.8001 - val_acc: 0.7135
Epoch 16/50
67/67 [==============================] - 44s 656ms/step - loss: 11.4271 - acc: 0.7835 - val_loss: 11.5731 - val_acc: 0.7057
Epoch 17/50
67/67 [==============================] - 44s 649ms/step - loss: 11.2133 - acc: 0.7765 - val_loss: 11.3231 - val_acc: 0.7240
Epoch 18/50
67/67 [==============================] - 45s 666ms/step - loss: 10.9761 - acc: 0.7807 - val_loss: 11.1079 - val_acc: 0.7135
Epoch 19/50
67/67 [==============================] - 46s 684ms/step - loss: 10.7761 - acc: 0.7765 - val_loss: 10.8868 - val_acc: 0.7292
Epoch 20/50
67/67 [==============================] - 45s 678ms/step - loss: 10.5286 - acc: 0.7793 - val_loss: 10.6785 - val_acc: 0.7292
Epoch 21/50
67/67 [==============================] - 45s 668ms/step - loss: 10.2911 - acc: 0.7807 - val_loss: 10.4539 - val_acc: 0.7266
Epoch 22/50
67/67 [==============================] - 44s 661ms/step - loss: 10.0571 - acc: 0.8036 - val_loss: 10.2370 - val_acc: 0.7318
Epoch 23/50
67/67 [==============================] - 44s 660ms/step - loss: 9.8697 - acc: 0.7695 - val_loss: 10.0265 - val_acc: 0.7292
Epoch 24/50
67/67 [==============================] - 44s 661ms/step - loss: 9.6392 - acc: 0.7876 - val_loss: 9.8480 - val_acc: 0.7057
Epoch 25/50
67/67 [==============================] - 45s 664ms/step - loss: 9.4508 - acc: 0.7812 - val_loss: 9.6062 - val_acc: 0.7214
Epoch 26/50
67/67 [==============================] - 44s 662ms/step - loss: 9.2311 - acc: 0.7868 - val_loss: 9.4190 - val_acc: 0.7188
Epoch 27/50
67/67 [==============================] - 44s 664ms/step - loss: 8.9801 - acc: 0.8059 - val_loss: 9.1784 - val_acc: 0.7422
Epoch 28/50
67/67 [==============================] - 44s 663ms/step - loss: 8.7932 - acc: 0.8012 - val_loss: 8.9944 - val_acc: 0.7161
Epoch 29/50
67/67 [==============================] - 45s 666ms/step - loss: 8.6165 - acc: 0.7961 - val_loss: 8.7903 - val_acc: 0.7083
Epoch 30/50
67/67 [==============================] - 45s 668ms/step - loss: 8.4344 - acc: 0.7839 - val_loss: 8.6090 - val_acc: 0.7292
Epoch 31/50
67/67 [==============================] - 45s 668ms/step - loss: 8.2147 - acc: 0.7979 - val_loss: 8.4374 - val_acc: 0.7188
Epoch 32/50
67/67 [==============================] - 45s 670ms/step - loss: 8.0108 - acc: 0.8072 - val_loss: 8.2485 - val_acc: 0.7318
Epoch 33/50
67/67 [==============================] - 44s 659ms/step - loss: 7.8510 - acc: 0.7933 - val_loss: 8.0716 - val_acc: 0.7396
Epoch 34/50
67/67 [==============================] - 44s 656ms/step - loss: 7.6727 - acc: 0.8068 - val_loss: 7.9088 - val_acc: 0.7344
Epoch 35/50
67/67 [==============================] - 44s 655ms/step - loss: 7.4570 - acc: 0.8194 - val_loss: 7.7259 - val_acc: 0.7109
Epoch 36/50
67/67 [==============================] - 44s 657ms/step - loss: 7.3005 - acc: 0.8082 - val_loss: 7.5322 - val_acc: 0.7474
Epoch 37/50
67/67 [==============================] - 44s 654ms/step - loss: 7.1286 - acc: 0.8092 - val_loss: 7.4001 - val_acc: 0.7474
Epoch 38/50
67/67 [==============================] - 44s 655ms/step - loss: 6.9418 - acc: 0.8203 - val_loss: 7.2326 - val_acc: 0.7370
Epoch 39/50
67/67 [==============================] - 44s 662ms/step - loss: 6.7808 - acc: 0.8246 - val_loss: 7.0668 - val_acc: 0.7292
Epoch 40/50
67/67 [==============================] - 44s 654ms/step - loss: 6.6156 - acc: 0.8283 - val_loss: 6.9394 - val_acc: 0.7214
Epoch 41/50
67/67 [==============================] - 44s 654ms/step - loss: 6.4837 - acc: 0.8278 - val_loss: 6.7808 - val_acc: 0.7318
Epoch 42/50
67/67 [==============================] - 44s 658ms/step - loss: 6.3314 - acc: 0.8190 - val_loss: 6.6353 - val_acc: 0.7240
Epoch 43/50
67/67 [==============================] - 44s 655ms/step - loss: 6.1902 - acc: 0.8274 - val_loss: 6.4697 - val_acc: 0.7266
Epoch 44/50
67/67 [==============================] - 44s 654ms/step - loss: 6.0323 - acc: 0.8282 - val_loss: 6.3272 - val_acc: 0.7370
Epoch 45/50
67/67 [==============================] - 44s 653ms/step - loss: 5.8951 - acc: 0.8190 - val_loss: 6.2076 - val_acc: 0.7578
Epoch 46/50
67/67 [==============================] - 44s 663ms/step - loss: 5.7585 - acc: 0.8218 - val_loss: 6.0836 - val_acc: 0.7396
Epoch 47/50
67/67 [==============================] - 44s 653ms/step - loss: 5.6190 - acc: 0.8245 - val_loss: 5.9363 - val_acc: 0.7266
Epoch 48/50
67/67 [==============================] - 44s 653ms/step - loss: 5.4875 - acc: 0.8385 - val_loss: 5.8219 - val_acc: 0.7370
Epoch 49/50
67/67 [==============================] - 44s 653ms/step - loss: 5.3653 - acc: 0.8306 - val_loss: 5.6780 - val_acc: 0.7370
Epoch 50/50
67/67 [==============================] - 44s 653ms/step - loss: 5.2427 - acc: 0.8227 - val_loss: 5.5714 - val_acc: 0.7240

Confusion Matrix
[[53  0  1  7  0  0]
 [ 0 57  5  0 14  0]
 [ 0  7 50  1  3  1]
 [ 6  3  1 71  3  6]
 [ 0  6  7 11 42  7]
 [ 1  4  1  5  6  5]]
Classification Report
              precision    recall  f1-score   support

   cardboard       0.88      0.87      0.88        61
       glass       0.74      0.75      0.75        76
       metal       0.77      0.81      0.79        62
       paper       0.75      0.79      0.77        90
     plastic       0.62      0.58      0.60        73
       trash       0.26      0.23      0.24        22

   micro avg       0.72      0.72      0.72       384
   macro avg       0.67      0.67      0.67       384
weighted avg       0.72      0.72      0.72       384
```


### Second long training period - 50 epochs
- L2 regularization: 0.1
- learning rate = 0.0001
- VGG-19 network last layers unfrozen

