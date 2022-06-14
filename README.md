# Mobile_robot_with_camera_Raspberry-Pi
Robot avoids obstacles using CNN model and USB-camera

My university project

/*------------------------------------------------------------*/

As a result of the work, I made a convolutional neural network model (using TensorFlow framework) for recognizing 2 classes of images, 70 in each, and apply it to the task of recognizing and detouring obstacles by a robot. The accuracy of the neural network on the test data set was 93.1%.
A PC was used to process neural network data, and a robot was used to control Raspberry Pi (due to lack of memory and an outdated version of the TensorFlow library on
Raspberry Pi). 
TCP/IP protocol was used for data exchange between PC and Raspberry, and to access Raspberry from a PC (to run the program that controls the robot) SSH protocol
As a result, it was possible to implement the recognition and detour of obstacles by the robot, but quite often the robot began to turn earlier than it was necessary.
Presumably, this problem is due to the fact that the frames from the Raspberry Pi came to the PC with a slight delay and, thus, real-time control of the robot was not fully provided.
