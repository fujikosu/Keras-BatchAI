from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(5, activation='softmax')(x)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 16

train_datagen = image.ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=0.2,
    zoom_range=0.2,
    shear_range=0.2)

valid_datagen = image.ImageDataGenerator()

test_datagen = image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        'image_split/training',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size) 

validation_generator = valid_datagen.flow_from_directory(
        'image_split/training',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size) 

# train the model on the new data for a few epochs
model.fit_generator(train_generator, steps_per_epoch=len(train_generator.classes) / batch_size, epochs=10, verbose=1, callbacks=None, validation_data=validation_generator, validation_steps=len(validation_generator.classes) / batch_size, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
