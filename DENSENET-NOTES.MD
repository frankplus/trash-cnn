## Transfer learning with DenseNet121

#### Architecture
```
DenseNet121 architecture
input_shape: (None, 224, 224, 3)
output_shape: (None, 7, 7, 1024)
_____________
max_pool
trainable: True
input_shape: (None, 7, 7, 1024)
output_shape: (None, 1024)
_____________
dense_1
trainable: True
input_shape: (None, 1024)
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
```

#### Short training period - 10 epochs
- L2 weights regularization: 0.01
- DenseNet121 all trainable
- Adam optimizer
- learning rate = 0.0001
- epsilon = default

![acc](plots/b_period_1.png)
![loss](plots/b_period_1_loss.png)

#### First long training period - 50 epochs
- L2 weights regularization: 0.01
- DenseNet121 all trainable
- SGD optimizer
- learning rate = 0.0001
- Nesterov momentum = 0.9

![acc](plots/b_period_2.png)
![loss](plots/b_period_2_loss.png)

#### Second long training period - 100 epochs
- L2 weights regularization: 0.05
- DenseNet121 all trainable
- SGD optimizer
- learning rate = 0.0001
- Nesterov momentum = 0.9

![acc](plots/b_period_3.png)
![loss](plots/b_period_3_loss.png)

#### Third long training period - 50 epochs
- L2 weights regularization: 0.05
- DenseNet121 all trainable
- SGD optimizer
- learning rate = 0.0001
- Nesterov momentum = 0.9

![acc](plots/b_period_4.png)
![loss](plots/b_period_4_loss.png)