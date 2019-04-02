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

