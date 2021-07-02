import cv2
import io
import socket
import struct
import time
import pickle
import zlib

import time
import RPi.GPIO as GPIO

# Declare the GPIO settings
GPIO.setmode(GPIO.BCM)

GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)
GPIO.setup(23, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('10.18.150.189', 9999))
connection = client_socket.makefile('wb')

cam = cv2.VideoCapture(0)

cam.set(3, 240) # width of frames
cam.set(4, 240) # height of frames

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
	ret, frame = cam.read()
	result, frame = cv2.imencode('Picture.png', frame, 	encode_param) # == cv2.imwrite()
	data = pickle.dumps(frame, 0)
	size = len(data)

	#print("{}: {}".format(img_counter, size))
	client_socket.sendall(struct.pack(">L", size) + data)
	img_counter += 1

	prob_blocked = client_socket.recv(1024) # receive response
	prob_blocked = struct.unpack('<1f', prob_blocked)

	# Turn right
	if prob_blocked[0] > 0.5:
		GPIO.output(17, GPIO.HIGH)
		GPIO.output(27, GPIO.LOW)
		GPIO.output(23, GPIO.LOW)
		GPIO.output(24, GPIO.LOW)
 		time.sleep(1)

	# Go forward
	GPIO.output(17, GPIO.HIGH)
	GPIO.output(27, GPIO.LOW)
	GPIO.output(23, GPIO.HIGH)
	GPIO.output(24, GPIO.LOW)
		
cam.release()
