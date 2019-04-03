from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from model import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

#parameters
img_width, img_height = 224, 224  # dimensions to which the images will be resized
trainset_dir = 'dataset-splitted/training-set'
testset_dir = 'dataset-splitted/test-set'
epochs = 100
batch_size = 32
num_classes = 6  #categories of trash
save_weights_file = "weights_save.h5"

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    trainset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)

test_generator = test_datagen.flow_from_directory(
    testset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=False)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = generate_transfer_model(input_shape, num_classes)


def fit(n_epochs):
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=n_epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator))

    model.save_weights('model_save.h5')

def print_layers():
    for layer in model.layers:
        print(layer.name)
        print("trainable: "+str(layer.trainable))
        print("input_shape: " + str(layer.input_shape))
        print("output_shape: " + str(layer.output_shape))
        print("_____________")

def load_weights():
    model.load_weights('weights_save.h5')

def classification_report():
    # Confution Matrix and Classification Report
    Y_pred = model.predict_generator(test_generator, len(test_generator))
    y_pred = np.argmax(Y_pred, axis=1)

    print('Confusion Matrix')
    conf_mat = confusion_matrix(test_generator.classes, y_pred)
    print(conf_mat)

    print('Classification Report')
    target_names = list(test_generator.class_indices.keys())
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))

