import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

import cv2
import os

import numpy as np

# Preprocessing

import glob

blocked = glob.glob('dataset/Blocked/*.*')
free = glob.glob('dataset/Free/*.*')

data = []
labels = []

for i in blocked:   
    image = tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size = (224,224))
    image = np.array(image)
    data.append(image)
    labels.append(0)
for i in free:   
    image = tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size = (224,224))
    image = np.array(image)
    data.append(image)
    labels.append(1)
    
data = np.array(data)
labels = np.array(labels)

# Split dataset 80/20 (x - data, y - labels)

X_train, x_test, Y_train, y_test = train_test_split(data, labels, test_size = 0.2,
                                                random_state = 42)

# Reshape labels as 2d-tensor (the first dim will is the batch dim and the second is the scalar label)

Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

# Make the CNN model

model = Sequential()
model.add(Conv2D(32, 3,padding="same", activation="relu", input_shape=(224,224,3))) # Convolutional layer
model.add(MaxPool2D()) # Pooling layer

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4)) # Randomly select 40% of the neurons and set their weights to 0 for one iteration
                        # to prevent overfitting

model.add(Flatten()) # Converts the pooled feature map to a single column 
                     # that is passed to the fully connected layer
    
model.add(Dense(128,activation="relu")) # Fully connected layer
model.add(Dense(2, activation="softmax")) # Output layer

model.summary()

# Compile the model (using Adam as optimizer and SparseCategoricalCrossentropy as the loss function)

opt = Adam(lr = 0.001)
model.compile(optimizer = opt, 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics = ['accuracy'])

# Train model

model_train = model.fit(X_train, Y_train, epochs = 6, validation_data = (x_test, y_test))

# Estimate model

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"test accuracy {acc*100} %")

# Save model

model.save('best_model.h5')
print('model saved')

model = tf.keras.models.load_model('best_model.h5')
pred = model.predict(x_test)
print(np.round(pred,2))
