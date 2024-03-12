import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True

def concatenate(layers):    
    #import pdb;pdb.set_trace()
    return tf.concat(layers, axis = 3)


def transferLayer(x, layer_number, dropout_rate):

    x = tf.compat.v1.layers.batch_normalization(x, axis = -1, center = True, scale = True, name = str(layer_number)+'bn')
    x = tf.nn.relu(x, name=str(layer_number)+'relu')
    x = tf.compat.v1.layers.dropout(x, rate=dropout_rate)          
    return x

def denseBlock(input_image, num_filters, kernel_size, density, block_number):
    concat_layers = []
    x = tf.compat.v1.layers.conv2d(input_image, padding='valid',filters = num_filters, kernel_size = kernel_size, activation = None)
    print("x shape before concat --->", x.shape)
    print("x type --->", type(x))
    concat_layers.append(x)
    #concat_layers.append(x)
    #import pdb;pdb.set_trace()
    for d in range(density-1):
        x = concatenate(concat_layers) 
        x = tf.compat.v1.layers.conv2d(x, name='conv_'+str(block_number)+str(d), padding = 'same', filters = num_filters, kernel_size = kernel_size, activation = None)
        x = tf.nn.relu(x, name='relu'+str(block_number)+str(d))
        concat_layers.append(x)
    x = concatenate(concat_layers)
    return x

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
    
    ip = tf.compat.v1.layers.conv2d(ip, name='conv1', filters = 16, kernel_size = 5, activation = None)
    ip = tf.compat.v1.layers.batch_normalization(ip)

    ip = denseBlock(ip, 32, 5, 3, 1)
    ip = transferLayer(ip, 2, 0.5)

    ip = denseBlock(ip, 32, 5, 3, 3)
    ip = transferLayer(ip, 4, 0.5)

    ip = tf.compat.v1.layers.flatten(ip)
    ip = tf.compat.v1.layers.dense(ip, name='fc1', units = 100, activation = None)
    ip = tf.compat.v1.layers.dense(ip, name='fc2', units = 10, activation = None)
    
    prLogits = ip
    prSoftMax = tf.nn.softmax(logits = prLogits)


    return prLogits, prSoftMax