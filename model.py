from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from keras import backend as K


def generate_model(input_shape, num_classes):
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