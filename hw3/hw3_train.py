#!/usr/bin/env python
# -- coding: utf-8 --
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras import backend as K
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
import sys
import numpy as np
import argparse
import time
import pandas as pd
from math import log, floor
from random import shuffle
#os.system('echo $CUDA_VISIBLE_DEVICES')
PATIENCE = 5 # The parameter is used for early stopping

# util
def show_train_history(train_history,train,validation):
    import matplotlib.pyplot as plt
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()

def group_encoder(vector) :
    onehot_encoder = OneHotEncoder(sparse=False) # the ans should be a dense matrix
    reshape_vector = vector.reshape(len(vector), 1)
    one_hot_matrix = onehot_encoder.fit_transform(reshape_vector)
    return  one_hot_matrix

def matrix_normalize(mat) :
    ans = mat.copy()
    shape = mat.shape
    for ii in range(0, shape[0], 1) :
        ans[ii,:] = preprocessing.scale(mat[ii,:])
        #ans[ii, :] = preprocessing.maxabs_scale(mat[ii, :])

    return ans

def LoadFile(train_data_path):
    #train_data_path = "/home/beth/ML/hw3/data/train.csv"
    label     = pd.read_csv(train_data_path).values[:, 0]
    data_raw  = pd.read_csv(train_data_path).values[:, 1]

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

    # check
    #diff = abs(data - data_new)
    #A = sum(diff.reshape(num_data* image_size * image_size))

    # process label
    label_one_hot = group_encoder(label)

    return data_nor, label_one_hot


def BuildModel():
    from keras.models import Sequential, Model, load_model
    from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
    from keras.layers.pooling import MaxPooling2D, AveragePooling2D
    from keras.layers.convolutional import Conv2D, ZeroPadding2D
    from keras.optimizers import SGD, Adam, Adadelta
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    #model = Sequential()
#
    ## conv 3 by 3 - 64
    #model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), input_shape=(48, 48, 1), padding='same', data_format="channels_last",
    #       activation="relu", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #       kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    #       bias_constraint=None))
#
    #model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same',
    #                 data_format="channels_last",
    #                 activation="relu", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    #                 bias_constraint=None))
#
    ##model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
    ## conv 3 by 3 - 128
    #model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
    #                 data_format="channels_last",
    #                 activation="relu", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    #                 bias_constraint=None))
#
    #model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
    #                 data_format="channels_last",
    #                 activation="relu", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    #                 bias_constraint=None))
#
    ## pooling
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
    ## conv 3 by 3 - 512
    #model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
    #                 data_format="channels_last",
    #                 activation="relu", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    #                 bias_constraint=None))
#
    #model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
    #                 data_format="channels_last",
    #                 activation="relu", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    #                 bias_constraint=None))
#
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
    ## Flatten
    #model.add(Flatten())
#
    ## fc-1024
    #model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.7))
#
    #model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.7))
#
    ## output
    #model.add(Dense(7, activation='softmax'))

    # build layer
    input_img = Input(shape=(48, 48, 1))
    '''
    先來看一下keras document 的Conv2D
    keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), 
        padding='valid', data_format=None, dilation_rate=(1, 1),
        activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None)
    '''
    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

    block4 = Conv2D(128, (3, 3), activation='relu')(block3)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

    block5 = Conv2D(128, (3, 3), activation='relu')(block4)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Flatten()(block5)

    fc1 = Dense(1024, activation='relu')(block5)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # loss function and compile
    #opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = Adam(lr=1e-3)
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


# load data
train_data_path = sys.argv[1]
train_data, train_label = LoadFile(train_data_path)

# build
model = BuildModel()


# train
filepath="./model/model_{epoch:02d}-{val_acc:.2f}.h5"
dirname = './model'
if not os.path.exists(dirname):
    os.mkdir(dirname)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

# check 5 epochs
early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max')

callbacks_list = [checkpoint, early_stop]
train_history=model.fit(x=train_data,
                        y=train_label,
                        validation_split=0.1,
                        epochs=100, batch_size=64,
                        verbose=2, callbacks=callbacks_list)

# save
model.save('./model.h5')
#show_train_history(train_history, 'acc', 'val_acc')
#show_train_history(train_history, 'loss', 'val_loss')
print("HI~~~")
