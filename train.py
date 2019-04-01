from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from model import generate_model

#parameters
img_width, img_height = 256, 256 # dimensions to which the images will be resized
dataset_dir = 'dataset-resized'
epochs = 50
batch_size = 32
num_classes = 6 #categories of trash

# this is the augmentation configuration we will use for training
dataset_generator = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.15)

train_generator = dataset_generator.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    subset="training")

test_generator = dataset_generator.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    subset="validation")

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = generate_model(input_shape, num_classes)


model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=len(test_generator))

model.save_weights('model_save.h5')