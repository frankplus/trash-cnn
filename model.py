from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization
from keras import backend as K
from keras import optimizers
from keras.applications import vgg19

# Generate model with the same architecture as used in the work by Mindy Yang and Gary Thung
def generate_trashnet_model(input_shape, num_classes):
    # create model
    model = Sequential()
    # add model layers
    model.add(Conv2D(96, kernel_size=11, strides=4, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Conv2D(256, kernel_size=5, strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Conv2D(384, kernel_size=3, strides=1, activation='relu'))
    model.add(Conv2D(384, kernel_size=3, strides=1, activation='relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation(lambda x: K.relu(x, alpha=1e-3)))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation(lambda x: K.relu(x, alpha=1e-3)))
    model.add(Dense(num_classes, activation="softmax"))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Generate model using the VGG-19 architecture pretrained with imagenet substituting the fully connected layer
def generate_transfer_model(input_shape, num_classes):
    model = Sequential()

    # imports the VGG19 model and discards the fc layer
    model.add(vgg19.VGG19(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_shape,
        pooling='max')) #using max global pooling, no flatten required

    # set VGG19 to fixed weights
    for layer in model.layers:
        layer.trainable = False

    # add fc layers
    model.add(Dense(256, activation="relu")) #TODO: confront relu vs leakyrelu ; add regularization
    model.add(Dropout(0.6))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation="softmax"))

    # compile model using accuracy to measure model performance and adam optimizer
    optimizer = optimizers.Adam() #TODO test different learning rate and epsilon
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
