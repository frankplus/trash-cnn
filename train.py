from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from model import generate_model
import matplotlib.pyplot as plt

#parameters
img_width, img_height = 256, 256 # dimensions to which the images will be resized
trainset_dir = 'dataset-splitted/training-set'
testset_dir = 'dataset-splitted/test-set'
epochs = 50
batch_size = 32
num_classes = 6 #categories of trash

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
    batch_size=batch_size)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = generate_model(input_shape, num_classes)

def fit(n_epochs):
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=n_epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator))

    model.save_weights('model_save.h5')

fit(epochs)