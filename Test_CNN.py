import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

import cv2
import PIL
import os

import numpy as np
import random

# Load the model

import glob

model = ['best_model.h5']

blocked = glob.glob('dataset/Blocked/*.*')
free = glob.glob('dataset/Free/*.*')

model = tf.keras.models.load_model('best_model.h5')

# Preprocessing

labels = []
f = []
b = []

for i in blocked:   
    image = tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size = (224,224))
    image = np.array(image)
    b.append(image)
    labels.append(0)
for i in free:   
    image = tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size = (224,224))
    image = np.array(image)
    f.append(image)
    labels.append(1)
    
b = np.array(b)
f = np.array(f)
labels = np.array(labels)

# Get confidence if the way is blocked

for image in f:
    
    image = np.asarray(image).astype('float32').reshape((-1,224,224,3))
    prob_blocked = model.predict(image)         
               
    print("Free dataset, blocked confidence: {}".format(np.round(prob_blocked[0][0],2)))

print('\n')
    
for image in b:
    
    image = np.asarray(image).astype('float32').reshape((-1,224,224,3))
    prob_blocked = model.predict(image)
               
    print("Blocked dataset, blocked confidence: {}".format(np.round(prob_blocked[0][0],2)))

# Check how USB-camera works

all_camera_idx_available = []

for camera_idx in range(10):
    cap = cv2.VideoCapture(camera_idx)
    if cap.isOpened():
        print(f'Camera index available: {camera_idx}')
        all_camera_idx_available.append(camera_idx)
        cap.release()

cap = cv2.VideoCapture(2)

def wait(delay):
    framecount = 0
    # capture and discard frames while the delay is not over
    while framecount < delay:
        cap.read()
        framecount += 1

while True:
    delay = 1
    wait(delay)
    # get and display next frame
    ret, img = cap.read()
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
