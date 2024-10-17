# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 19:31:11 2024

@author: Rasika
"""

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and preprocess the datasets
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

import tensorflow as tf
from tensorflow.keras import layers, models

def MobileNet(input_shape=(224, 224, 3), num_classes=1000):
    model = models.Sequential()

    # Convolution Layer
    model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())

    # Depthwise Separable Convolutions
    def depthwise_separable_conv(filters, kernel_size=(3, 3), strides=(1, 1)):
        model.add(layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters, kernel_size=(1, 1), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())

    # First block
    depthwise_separable_conv(64)
    
    # Downsample
    depthwise_separable_conv(128, strides=(2, 2))
    depthwise_separable_conv(128)
    
    # Downsample
    depthwise_separable_conv(256, strides=(2, 2))
    depthwise_separable_conv(256)
    
    # Downsample
    depthwise_separable_conv(512, strides=(2, 2))

    # Repeating block
    for _ in range(5):
        depthwise_separable_conv(512)
    
    # Downsample
    depthwise_separable_conv(1024, strides=(2, 2))
    depthwise_separable_conv(1024)

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())
    
    # Fully Connected Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Instantiate the model
mobilenet = MobileNet(input_shape=(224, 224, 3), num_classes=1000)
mobilenet.summary()
mobilenet.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load the base MobileNet model with pre-trained ImageNet weights
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Add global average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer with 1 unit (binary classification)
x = Dense(1, activation='sigmoid')(x)

# Create the new model
mobilenet = Model(inputs=base_model.input, outputs=x)

# Freeze the layers of the base model to avoid retraining them
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
mobilenet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = mobilenet.fit(train_ds, epochs=6, validation_data=validation_ds)

# Visualize accuracy
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()

import cv2
testimg = cv2.imread(r"C:\Users\91994\Downloads\cateq256-256.jpg")
image = tf.image.resize(testimg, (256, 256))  # Resize image to 256x256
image = tf.expand_dims(image, axis=0)  # Add batch dimension
predictions =mobilenet.predict(image) 
print(predictions)
testimg1 = cv2.imread(r"C:\Users\91994\Downloads\dog.png")
image1 = tf.image.resize(testimg1, (256, 256))  # Resize image to 256x256
image1= tf.expand_dims(image1, axis=0)  # Add batch dimension
predictions1 =mobilenet.predict(image1) 
print(predictions1)