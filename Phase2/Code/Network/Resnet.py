import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def split_paths(input_image, num_filters, kernel_size, block_number, layer_number):

    i = tf.compat.v1.layers.conv2d(inputs =input_image, name=str(layer_number)+'conv', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    i  = tf.compat.v1.layers.batch_normalization(inputs = i ,axis = -1, center = True, scale = True, name = str(layer_number)+'bn')

    f_i = tf.compat.v1.layers.conv2d(input_image, name=str(block_number)+'blk_conv1', padding = 'same', filters = num_filters, kernel_size = kernel_size, activation = None)
    f_i = tf.compat.v1.layers.batch_normalization(f_i, name=str(block_number)+'blk_batchnorm1')
    f_i = tf.nn.relu(f_i, name=str(block_number)+'blk_relu1')

    f_i = tf.compat.v1.layers.conv2d(f_i, name=str(block_number)+'blk_conv2', padding = 'same',filters = num_filters, kernel_size = kernel_size, activation = None)
    f_i = tf.compat.v1.layers.batch_normalization(f_i, name=str(block_number)+'blk_batchnorm2')

    h_i = tf.math.add(i, f_i)
    h_i = tf.nn.relu(h_i, name='relu'+str(layer_number))
    return h_i


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
    
    ip = split_paths(ip, num_filters=32, kernel_size=5, block_number=1, layer_number=2)
    ip = split_paths(ip, num_filters=64, kernel_size=5, block_number=2, layer_number=3)

    ip = tf.compat.v1.layers.flatten(ip)
    ip = tf.compat.v1.layers.dense(ip, name='fullyconnected1', units = 100, activation = None)
    ip = tf.compat.v1.layers.dense(ip, name='fullyconnected2', units = 10, activation = None)
    
    prLogits = ip
    prSoftMax = tf.nn.softmax(logits = prLogits)


    return prLogits, prSoftMax