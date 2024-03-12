import tensorflow as tf
import sys
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True


def cardinality_block(i, num_filters1, num_filters2, kernal_size1, kernal_size2, b, n,c):
    i = tf.compat.v1.layers.conv2d(i, name= 'conv1_'+str(b)+str(n)+str(c), padding='same',filters = num_filters1, kernel_size = kernal_size1, activation = None)
    i = tf.compat.v1.layers.batch_normalization(i,axis = -1, center = True, scale = True, name = 'bn1_'+str(b)+str(n)+str(c))
    i = tf.nn.relu(i, name = 'reu1'+str(b)+str(n)+str(c))

    i = tf.compat.v1.layers.conv2d(i, name= 'conv2_'+str(b)+str(n)+str(c), padding='same',filters = num_filters2, kernel_size = kernal_size2, activation = None)
    i = tf.compat.v1.layers.batch_normalization(i,axis = -1, center = True, scale = True, name ='bn2_'+str(b)+str(n)+str(c))
    return i

def residualBlock(input_image, num_filters, num_filters1, num_filters2, kernal_size, kernal_size1, kernal_size2, cardinality, block_number, layer_number):

    i = tf.compat.v1.layers.conv2d(inputs =input_image, name='conv_'+str(block_number)+str(layer_number), padding='same',filters = num_filters, kernel_size = kernal_size, activation = None)
    i  = tf.compat.v1.layers.batch_normalization(inputs = i ,axis = -1, center = True, scale = True, name = 'bn_'+str(block_number)+str(layer_number))
    i_store = i

    i_merged = i_store

    for c in range(cardinality):
        i_path = cardinality_block(i, num_filters1, num_filters2, kernal_size1, kernal_size2, block_number, layer_number, c)
        i_merged = tf.math.add(i_merged, i_path)

    i = tf.nn.relu(i_merged, name='relu_'+str(block_number)+str(layer_number))
    return i


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
    
    ip = residualBlock(input_image = ip, num_filters = 16, num_filters1 = 16, num_filters2 = 16, kernal_size=5, kernal_size1=5, kernal_size2=5, cardinality=6, block_number=1, layer_number=1)
    ip = residualBlock(input_image = ip, num_filters = 16, num_filters1 = 16, num_filters2 = 16, kernal_size=5, kernal_size1=5, kernal_size2=5, cardinality=6, block_number=2, layer_number=2)

    ip = tf.compat.v1.layers.flatten(ip)
    ip = tf.compat.v1.layers.dense(ip, name='fc1', units = 100, activation = None)
    ip = tf.compat.v1.layers.dense(ip, name='fc2', units = 10, activation = None)
    
    prLogits = ip
    prSoftMax = tf.nn.softmax(logits = prLogits)


    return prLogits, prSoftMax