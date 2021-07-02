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
from PIL import Image
import os
import io

import numpy as np
import random
import glob

import socket
import sys
import pickle
import struct
import zlib
from io import BytesIO

# Load the model

model = tf.keras.models.load_model('best_model.h5')

# Connect to RPi and exchange data

HOST = '10.18.150.189'
PORT = 9999

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

s.bind((HOST,PORT))
s.listen(10)

conn, addr = s.accept()

data = b""
payload_size = struct.calcsize(">L")
#print("payload_size: {}".format(payload_size))
while True:
    while len(data) < payload_size:
        #print("Recv: {}".format(len(data)))
        data += conn.recv(4096)

    #print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    #print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:] # get bytes
    
    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes") # convert bytes into numpy array
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR) # read data and convert it to image format (also np.array)
    
    #cv2.imshow('Image',image)
    
    width = 224
    height = 224
    dim = (width, height)
    image = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) # (224,224,3)
    image = np.asarray(image).astype('float32').reshape((-1,3))
    image = image.reshape(50176,3).reshape(-1,224,224,3)
    
    prob_blocked = model.predict(image)
    print(prob_blocked[0][0])
    
    prob_blocked = struct.pack('<1f', prob_blocked[0][0])
    conn.send(prob_blocked)
    
    cv2.waitKey(1)
