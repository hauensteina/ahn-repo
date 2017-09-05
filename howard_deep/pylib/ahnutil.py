# /********************************************************************
# Filename: ahnutil.py
# Author: AHN
# Creation Date: Aug 27, 2017
# **********************************************************************/
#
# Various utility funcs
#

from __future__ import division,print_function

from pdb import set_trace as BP
import os,sys,re,json
import numpy as np
import keras.preprocessing.image as kp
from keras.utils.np_utils import to_categorical


# Return iterators to get batches of images from a folder.
# Example:
'''
    batches = get_batches('data', batch_size=2)
    nextbatch = batches['train_batches'].next() # gets two random images
'''
# [[image1,image2],[class1,class2]]
# Classes are one-hot encoded.
# If class_mode==None, just return  [image1,image2,...]
# WARNING: The images must be in *subfolders* of  path/train and path/valid.
#-----------------------------------------------------------------------
def get_batches(path,
                gen=kp.ImageDataGenerator(),
                shuffle=True,
                batch_size=4,
                class_mode='categorical',
                target_size=(224,224),
                color_mode='grayscale'):
    train_path = path + '/' + 'train'
    valid_path = path + '/' + 'valid'
    train_batches = gen.flow_from_directory(train_path,
                                            target_size=target_size,
                                            class_mode=class_mode,
                                            shuffle=shuffle,
                                            batch_size=batch_size,
                                            color_mode=color_mode)
    valid_batches = gen.flow_from_directory(valid_path,
                                            target_size=target_size,
                                            class_mode=class_mode,
                                            shuffle=shuffle,
                                            batch_size=batch_size,
                                            color_mode=color_mode)
    res = {'train_batches':train_batches, 'valid_batches':valid_batches}
    return res

# One-hot encode a list of integers
# (1,3,2) ->
# array([[ 0.,  1.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.],
#        [ 0.,  0.,  1.,  0.]])
#-------------------------------------
def onehot(x):
    return to_categorical(x)


# Get means and stds by channel(color) from array of imgs
#----------------------------------------------------------
def get_means_and_stds(images):
    mean_per_channel = images.mean(axis=(0,2,3), keepdims=1)
    std_per_channel  = images.std(axis=(0,2,3), keepdims=1)
    return mean_per_channel, std_per_channel

# Subtract supplied mean from each channel, divide by sigma
#----------------------------------------------------------
def normalize(images, mean_per_channel, std_per_channel):
    images -= mean_per_channel
    images /= std_per_channel

# Get all images below a folder into one huge numpy array
# WARNING: The images must be in *subfolders* of path/train and path/valid.
#-----------------------------------------------------------
def get_data(path, target_size=(224,224), color_mode='grayscale'):
    batches = get_batches(path,
                          shuffle=False,
                          batch_size=1,
                          class_mode=None,
                          target_size=target_size,
                          color_mode=color_mode
    )
    train_data =  np.concatenate([batches['train_batches'].next() for i in range(batches['train_batches'].samples)])
    valid_data =  np.concatenate([batches['valid_batches'].next() for i in range(batches['valid_batches'].samples)])
    res = {'train_data':train_data.astype(float), 'valid_data':valid_data.astype(float)}
    return res

# Get arrays with meta info matching the order of the images
# returned by get_data().
# We take the class from a json file, not a folder name.
# Example:
# images = get_data(....)
# image = images['train'][42]
# meta =  get_meta(...)
# one_hot_class_for_image = meta['train_classes_hot'][42]
#----------------------------------------------------------
def get_meta(path):
    batches = get_batches(path, shuffle=False, batch_size=1)
    train_batches = batches['train_batches']
    valid_batches = batches['valid_batches']

    train_classes=[]
    for idx,fname in enumerate(train_batches.filenames):
        jf = path + '/train/' + os.path.splitext(fname)[0]+'.json'
        j  = json.load(open(jf, 'r'))
        train_classes.append(int(j['class']))
    train_classes_hot = onehot(train_classes)

    valid_classes=[]
    for idx,fname in enumerate(valid_batches.filenames):
        jf =  path + '/valid/' + os.path.splitext(fname)[0]+'.json'
        j = json.load(open(jf, 'r'))
        valid_classes.append(int(j['class']))
    valid_classes_hot = onehot(valid_classes)

    res = {
        'train_classes':train_classes,
        'train_classes_hot':train_classes_hot,
        'train_filenames':train_batches.filenames,
        'valid_classes':valid_classes,
        'valid_classes_hot':valid_classes_hot,
        'valid_filenames':valid_batches.filenames
    }
    return res

# Get arrays with expected output matching the order of the images
# returned by get_data().
# Returns the value found for key in each json file.
# Example:
# images = get_data(....)
# image = images['train'][42]
# meta =  get_output_by_key(...)
# meta_for_image = meta['train_output'][42]
#----------------------------------------------------------
def get_output_by_key(path,key):
    batches = get_batches(path, shuffle=False, batch_size=1)
    train_batches = batches['train_batches']
    valid_batches = batches['valid_batches']

    train_output=[]
    for idx,fname in enumerate(train_batches.filenames):
        jf = path + '/train/' + os.path.splitext(fname)[0]+'.json'
        j  = json.load(open(jf, 'r'))
        train_output.append(j[key])

    valid_output=[]
    for idx,fname in enumerate(valid_batches.filenames):
        jf =  path + '/valid/' + os.path.splitext(fname)[0]+'.json'
        j = json.load(open(jf, 'r'))
        valid_output.append(j[key])

    res = {
        'train_output':train_output,
        'train_filenames':train_batches.filenames,
        'valid_output':valid_output,
        'valid_filenames':valid_batches.filenames
    }
    return res
