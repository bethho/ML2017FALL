#!/usr/bin/env python
# -- coding: utf-8 --
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.models import Sequential
from keras.utils import np_utils
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
from keras import backend as K
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
import numpy as np
import argparse
import time
import pandas as pd
from math import log, floor
from random import shuffle
#os.system('echo $CUDA_VISIBLE_DEVICES')
PATIENCE = 5 # The parameter is used for early stopping


def matrix_normalize(mat) :
    ans = mat.copy()
    shape = mat.shape
    for ii in range(0, shape[0], 1) :
        ans[ii,:] = preprocessing.scale(mat[ii,:])
        #ans[ii, :] = preprocessing.maxabs_scale(mat[ii, :])

    return ans

def LoadTestFile(test_data_path):
    #test_data_path = "/home/beth/ML/hw3/data/test.csv"

    data_raw  = pd.read_csv(test_data_path).values[:, -1]

    #process data
    num_data = data_raw.shape[0]
    image_size = 48

    #data = []
    data_new = np.zeros( (num_data, image_size * image_size) )
    for i in range(num_data ):
        #data.append(data_raw[i])
        #data[-1]      = np.fromstring(data_raw[i], dtype=float, sep=' ').reshape((48, 48, 1))
        data_new[i,:] = np.fromstring(data_raw[i], dtype=float, sep=' ')

    # data = np.asarray(data)

    # normailzation
    #data_nor = data_new
    data_nor = matrix_normalize(data_new)
    data_nor = data_nor.reshape(num_data, image_size, image_size, 1)

    return data_nor


def ind_max(A):
    a = 0
    max = A[0]
    for i in range(len(A)):
        if A[i] > max:
            a = i
            max = A[i]
    return a
test_data_path = sys.argv[1]
test_pixels = LoadTestFile(test_data_path)
model = load_model('./model_simple_0.59.h5')
prediction = model.predict(test_pixels)
labels = np.zeros((prediction.shape[0],1))

for i in range(prediction.shape[0]):
    labels[i] = ind_max(prediction[i])
#output_path = "./result/predit_1.csv"
output_path = sys.argv[2]
dirname = os.path.dirname(output_path)
if not os.path.exists(dirname):
    os.mkdir(dirname)
with open(output_path, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(labels):
        f.write('%d,%d\n' %(i, v))

