"""# Importing all the pre-requisite libraries

---
"""

from keras.applications.resnet50 import ResNet50
from sklearn import metrics
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import keras
import cv2
import shutil
import tensorflow as tf
from keras.layers import LeakyReLU
from keras.metrics import categorical_accuracy
from keras.utils import plot_model, to_categorical
from keras.models import Model, load_model, Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

"""# Neural Network Modelling
Using **Resnet50** as the base model for the neural network framework.
Then adding 3 fully connected layers of 1024 neurons with *relu* activation function and adding the last output layer using *softmax* activation with 29 nuerons.


---
"""

img_height, img_width = (200, 200)
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(200, 200, 3))
for layer in base_model.layers:
    layer.trainable = False
num_classes = 29
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
# x = Dropout(0.3)(x)
x = Dense(1024, activation='relu')(x)
# x = Dropout(0.3)(x)
x = Dense(1024, activation='relu')(x)
# x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

"""# Learning Algorithm
Using adam optimizer with loss as categorical crossentropy and printing out accuracy as metrics while learning.

*Define the directory if you are running notebook on local machine*


---


## Train 1
First the model is trained on the Resnet for 10 epochs and batch size of 32

---
"""

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 10
BS = 32
aug = ImageDataGenerator(rotation_range=5, zoom_range=0.2, rescale=1./255,
                         width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.18,
                         horizontal_flip=False, fill_mode="nearest")

# directory for image as data
directory = "/content/asl_alphabet_train"

image_generator = aug.flow_from_directory(directory, target_size=(200, 200),
                                          class_mode='categorical', batch_size=32)

H = model.fit_generator(
    image_generator, steps_per_epoch=2718, epochs=EPOCHS).cuda()

for layer in model.layers:
    layer.trainable = True

"""## Train 2
Then it is trained on the 3 additional fully connected layers for 2 epochs and batch size of 32

---
"""

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 2
BS = 32
aug = ImageDataGenerator(rotation_range=5, zoom_range=0.2, rescale=1./255,
                         width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.18,
                         horizontal_flip=False, fill_mode="nearest")

# directory for image as data
directory = "/content/asl_alphabet_train"

image_generator = aug.flow_from_directory(directory, target_size=(200, 200),
                                          class_mode='categorical', batch_size=32)

H2 = model.fit_generator(image_generator, steps_per_epoch=2718, epochs=EPOCHS)

"""## Final steps
Saving the model and printing history of first and second train


---
"""

model.save("./resnet1.h5")

H.history

H2.history
