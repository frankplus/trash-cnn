# trash-cnn

This is a convolutional neural network implementation for garbage image classification

The dataset we are using is from the work by Gary Thung and Mindy Yang and can be found [here](https://github.com/garythung/trashnet/) \
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

### Execute
I suggest to run using the -i interpreter option to run commands on terminal 
after execution such as `fit(epochs)` or `print_layers()`

### The model
For this project I tried to take advantage of the transfer learning technique which turns out to be very promising.
For the notes about the learning with different parameters see: [densenet](DENSENET-NOTES.MD) and [VGG-19](VGG19_NOTES.MD)