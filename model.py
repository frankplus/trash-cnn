from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization
from keras import backend as K
from keras import optimizers, regularizers, Model
from keras.applications import vgg19, densenet

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


# Generate model using a pretrained architecture substituting the fully connected layer
def generate_transfer_model(input_shape, num_classes):

    # imports the pretrained model and discards the fc layer
    base_model = densenet.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_shape,
        pooling='max') #using max global pooling, no flatten required

    # train only the top layers, i.e. freeze all convolutional layers
    # for layer in base_model.layers:
    #     layer.trainable = False

    # unfreeze last convolutional block
    # train_last_conv_layer = True
    # if train_last_conv_layer:
    #     n_layers_unfreeze = 7*16+3
    #     for layer in base_model.layers[-n_layers_unfreeze:]:
    #         layer.trainable = True

    # add fc layers
    x = base_model.output
    #x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.6)(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # compile model using accuracy to measure model performance and adam optimizer
    optimizer = optimizers.Adam(lr=0.001)
    #optimizer = optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model