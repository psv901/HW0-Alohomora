#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
import os
import glob
#import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Resnet import CIFAR10Model
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from io import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import csv
from sklearn.preprocessing import StandardScaler

# Don't generate pyc codes
sys.dont_write_bytecode = True

    
def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    BasePath - Path to CIFAR10 folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain)-1)
        
        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.png'   
        ImageNum += 1
    	
    	##########################################################
    	# Add any standardization or data augmentation here!
    	##########################################################

        I1 = np.float32(cv2.imread(RandImageName))
        
        p = np.mean(I1, axis=(1,2), keepdims=True)
        q = np.std(I1, axis=(1,2), keepdims=True)
        augmented_image = (I1 - p) / (q + 0.0001)
        
        Label = convertToOneHot(TrainLabels[RandIdx], 10)

        # Append All Images and Mask
        I1Batch.append(augmented_image)
        LabelBatch.append(Label)
        
    return I1Batch, LabelBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    

def TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, plotPath):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to CIFAR10 folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """      
    # Predict output with forward pass
    prLogits, prSoftMax = CIFAR10Model(ImgPH, ImageSize, MiniBatchSize)

    with tf.name_scope('Loss'):
        ###############################################
        # Fill your loss function of choice here!
        ###############################################
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=LabelPH, logits=prLogits)
        loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope('Accuracy'):
        prSoftMaxDecoded = tf.argmax(prSoftMax, axis=1)
        LabelDecoded = tf.argmax(LabelPH, axis=1)
        Acc = tf.reduce_mean(tf.cast(tf.math.equal(prSoftMaxDecoded, LabelDecoded), dtype=tf.float32))
        
    with tf.name_scope('Adam'):
    	###############################################
    	# Fill your optimizer of choice here!
    	###############################################
        Optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    tf.summary.scalar('Accuracy', Acc)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.compat.v1.train.Saver(max_to_keep=100)
    
    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.compat.v1.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.compat.v1.summary.FileWriter(LogsPath, graph=tf.get_default_graph())

        loss_epochs = []
        accuracy_epochs = []
        epochs = []

        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            loss_iterations = []
            accuracy_iterations = []
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                I1Batch, LabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize)
                FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch}
                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
                AccThisBatch = sess.run(Acc, feed_dict=FeedDict)
                
                loss_iterations.append(LossThisBatch)
                accuracy_iterations.append(AccThisBatch*100)
                # Save checkpoint every some SaveCheckPoint's iterations
                if PerEpochCounter % SaveCheckPoint == 0:
                    # Save the Model learnt in this epoch
                    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print('\n' + SaveName + ' Model Saved...')

                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()
            loss_epochs.append(np.mean(loss_iterations))
            accuracy_epochs.append(np.mean(accuracy_iterations))
            epochs.append(Epochs)

            print("loss over epoch = ", np.mean(loss_iterations))
            print("Accuraccy over epoch = ", np.mean(accuracy_iterations))

            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved...')
        plt.subplots(1, 2, figsize=(15,15))
        
        plt.subplot(1, 2, 1) #loss
        plt.plot(epochs, loss_epochs)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.xlim((0,NumEpochs))


        plt.subplot(1, 2, 2) #Accuracy
        plt.plot(epochs, accuracy_epochs)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.xlim((0,NumEpochs))
        plt.show()

        plt.savefig(plotPath + "training.png")
            

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/Users/psv/Downloads/YourDirectoryID_hw0/Phase2/CIFAR10/', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/CIFAR10')
    Parser.add_argument('--CheckPointPath', default='/Users/psv/Downloads/YourDirectoryID_hw0/Phase2/Code/Checkpoints/resnet/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=10, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=128, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='/Users/psv/Downloads/YourDirectoryID_hw0/Phase2/Code/Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    Parser.add_argument('--Plotpath', default='/Users/psv/Downloads/YourDirectoryID_hw0/Phase2/Code/plots/resnet/', help='Path to save plots')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    plotPath = Args.Plotpath

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)
    TrainLabels = list(TrainLabels)



    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.compat.v1.placeholder(tf.float32, shape=(None, ImageSize[0], ImageSize[1], ImageSize[2]))
    LabelPH = tf.compat.v1.placeholder(tf.float32, shape=(None, NumClasses)) # OneHOT labels
    
    TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, plotPath)
        
    
if __name__ == '__main__':
    main()
 
