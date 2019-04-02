# trash-cnn

This is a convolutional neural network implementation for garbage image classification

The dataset we are using can be found [here](https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip) \
I have splitted the dataset into train and test set with the following directory structure:
```
.
+-- dataset-splitted
|   +-- test-set
|   |   +-- cardboard
|   |   +-- glass
|   |   +-- ...
|   +-- training-set
|   |   +-- cardboard
|   |   +-- glass
|   |   +-- ...
```

## Dependencies
- tensorflow (or tensorflow-gpu if using GPU for training)
- keras

### execute
I suggest to run using the -i interpreter option to run commands on terminal 
after execution such as `fit(epochs)` or `print_layers()`