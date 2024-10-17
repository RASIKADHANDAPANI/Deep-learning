# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 20:26:22 2024

@author: Rasika
"""

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import layers, models

train_ds = keras.utils.image_dataset_from_directory(
    directory=r'F:\semII\dog vs cat\dataset\training_set',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory=r'F:\semII\dog vs cat\dataset\test_set',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)
def process(image,label):
  image = tf.cast(image/255. ,tf.float32)
  return image,label
train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)



def AlexNet(input_shape=(256, 256, 3), num_classes=1):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.BatchNormalization())
    
   
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(num_classes, activation='sigmoid'))  # For binary classification

    return model

alexnet = AlexNet(input_shape=(256, 256, 3), num_classes=1)
alexnet.summary()
alexnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = alexnet.fit(train_ds, epochs=6, validation_data=validation_ds)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()

# Image prediction
import cv2
testimg = cv2.imread(r"C:\Users\91994\Downloads\cateq256-256.jpg")
image = tf.image.resize(testimg, (256, 256))  # Resize image to 256x256
image = tf.expand_dims(image, axis=0)  # Add batch dimension
predictions = alexnet.predict(image) 
print(predictions)

testimg1 = cv2.imread(r"C:\Users\91994\Downloads\dog.png")
image1 = tf.image.resize(testimg1, (256, 256))  # Resize image to 256x256
image1 = tf.expand_dims(image1, axis=0)  # Add batch dimension
predictions1 = alexnet.predict(image1) 
print(predictions1)
