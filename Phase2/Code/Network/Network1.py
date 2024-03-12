import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################
    ip = Img
    ip = tf.compat.v1.layers.conv2d(ip, name='convoution1', filters = 16, kernel_size = 5, activation = None)
    ip = tf.compat.v1.nn.relu(ip, name='relu1')

    ip = tf.compat.v1.layers.conv2d(ip, name='convoution1', filters = 16, kernel_size = 5, activation = None)
    ip = tf.compat.v1.nn.relu(ip, name='relu1')

    ip = tf.compat.v1.layers.conv2d(ip, name='convolution3', filters = 32, kernel_size = 3, activation = None)
    ip = tf.nn.relu(ip, name='relu2')

    ip = tf.compat.v1.layers.flatten(ip)
    ip = tf.compat.v1.layers.dense(ip, name='fullyconnected1', units = 100, activation = None)

    ip = tf.compat.v1.layers.dense(ip, name='fullyconnected2', units = 10, activation = None)

    prLogits = ip
    prSoftMax = tf.nn.softmax(logits = prLogits)  

    return prLogits, prSoftMax
