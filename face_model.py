# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:19:00 2020

@author: rashmibh
"""

# Python program to create 
# Image Classifier using CNN 
  
# Importing the required libraries 

import matplotlib.pyplot as plt
import numpy as np

TRAIN_DIR = 'dataset/training_set/'
TEST_DIR = 'dataset/test_set/'
IMG_SIZE = 64
BATCH_SIZE=16
NO_CLASSES=0
  
#Generate Training and Test sets
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range=40,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range=40)

training_set = train_datagen.flow_from_directory(TRAIN_DIR,
                                                 target_size = (IMG_SIZE, IMG_SIZE),
                                                 color_mode="grayscale",
                                                 batch_size = BATCH_SIZE,
                                                  class_mode="categorical",
                                                  shuffle=True)

test_set = test_datagen.flow_from_directory(TEST_DIR,
                                            target_size = (IMG_SIZE, IMG_SIZE),
                                            color_mode="grayscale",
                                            batch_size = BATCH_SIZE,
                                            class_mode="categorical",
                                            shuffle=False)


NO_CLASSES=len(training_set.class_indices)

np.save('class_indices_saved_celeb', training_set.class_indices)

'''Running the training and the testing in the dataset for our model'''
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout,BatchNormalization

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 1, input_shape = (64, 64, 1), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.25))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 1, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.25))


# Adding 3rd Convolution layer
classifier.add(Convolution2D(64,(3,1), padding='same', activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

# Step 3 - Flattening
classifier.add(Flatten()) 

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = NO_CLASSES, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()

hist=classifier.fit_generator(training_set,
                         nb_epoch = 100,
                         validation_data = test_set)

classifier.save('model.facedetect_celeb')

 # Get training and validation loss histories
train_loss = hist.history['loss']
validation_loss = hist.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(train_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, train_loss, 'r--')
plt.plot(epoch_count, validation_loss, 'g--')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

 # Get training and validation loss histories
train_accu = hist.history['accuracy']
validation_accu= hist.history['val_accuracy']

# Visualize loss history
plt.plot(epoch_count, train_accu, 'r--')
plt.plot(epoch_count, validation_accu, 'g--')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show();


classifier_json = classifier.to_json()
with open("model.facedetect_celeb.json", "w") as json_file:
  json_file.write(classifier_json)
classifier.save_weights("model.facedetect_celeb.weights")



