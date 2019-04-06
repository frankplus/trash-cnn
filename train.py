from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model import *
#from google.colab import drive
import tensorflow as tf
import seaborn as sn
import pandas as pd

# Mount Google Drive
#drive.mount('/gdrive')

# parameters
img_width, img_height = 224, 224  # dimensions to which the images will be resized
n_epochs = 50
batch_size = 32
num_classes = 6  # categories of trash

#project_dir = '/gdrive/My Drive/trash-cnn/'
project_dir = ''
trainset_dir = project_dir + 'dataset-splitted/training-set'
testset_dir = project_dir + 'dataset-splitted/test-set'
load_weights_file = project_dir + 'weights_save_densenet121_val_acc_86.0.h5'
save_weights_file = project_dir + 'weights_save_4.h5'

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

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


def load_weights():
    model.load_weights(load_weights_file)
    print("Weights loaded")


def fit(n_epochs):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=n_epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator))

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save_weights(save_weights_file)


def print_layers():
    for layer in model.layers:
        print(layer.name)
        print("trainable: " + str(layer.trainable))
        print("input_shape: " + str(layer.input_shape))
        print("output_shape: " + str(layer.output_shape))
        print("_____________")


def print_classification_report():
    # Confution Matrix and Classification Report
    Y_pred = model.predict_generator(test_generator, len(test_generator))
    y_pred = np.argmax(Y_pred, axis=1)

    print('Classification Report')
    target_names = list(test_generator.class_indices.keys())
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))

    print('Confusion Matrix')
    conf_mat = confusion_matrix(test_generator.classes, y_pred)
    df_cm = pd.DataFrame(conf_mat, index=target_names, columns=target_names)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

#save keras model and convert it into tflite model
def save_model():
    # Save tf.keras model in HDF5 format.
    keras_file = "keras_model.h5"
    model.save(keras_file)

    # Convert to TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    print("saved")


#print_layers()
load_weights()
#fit(n_epochs)
print_classification_report()
#save_model()