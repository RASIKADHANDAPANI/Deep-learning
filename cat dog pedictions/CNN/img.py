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

def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=3
)

loss, accuracy = model.evaluate(validation_ds)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

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
import numpy as np


# Load and resize the image
testimg = cv2.imread(r"C:\Users\91994\Downloads\dog.png")
testimg = cv2.cvtColor(testimg, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
testimg_resized = tf.image.resize(testimg, [256, 256])  # Resize to match model input

# Normalize the image
testimg_resized = testimg_resized / 255.0

# Add batch dimension
testimg_batch = tf.expand_dims(testimg_resized, axis=0)  # Shape (1, 256, 256, 3)

# Make prediction
predictions = model.predict(testimg_batch)
print(predictions)

# Display the image
plt.imshow(testimg_resized.numpy())  # Convert to numpy for matplotlib
plt.title('Test Image')
plt.show()

