# -*- coding: utf-8 -*-
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

#from _future_ import print_function
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import sys, argparse, os
import keras
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
sess = tf.Session(config=config)
K.set_session(sess)
#import tensorflow as tf
#from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Convolution2D ,Flatten ,Dense, MaxPooling2D, BatchNormalization, LSTM, GRU, Bidirectional, TimeDistributed, concatenate, Lambda
from keras.utils import np_utils
from keras.optimizers import SGD ,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
from keras import optimizers
from keras.layers.embeddings import Embedding

parser = argparse.ArgumentParser(description='MF')
parser.add_argument('--embedding_dim', default=256, type=int)
parser.add_argument('--user_path', default='data/users.csv')
parser.add_argument('--movie_path', default='data/movies.csv')
parser.add_argument('--test_path', default='data/test.csv')
parser.add_argument('--result_path', default='result_best.csv')
args = parser.parse_args()
print(args.embedding_dim)
def normalize(X):
    # Feature normalization with train and test X
    mu1 = (sum(X) / X.shape[0])
    sigma1 = np.std(X, axis=0)
    mu = np.tile(mu1, (1, X.shape[0]))
    sigma = np.tile(sigma1, (1, X.shape[0]))
    X_normed = (X - mu) / sigma
    return X_normed, mu1, sigma1

#EMBEDDING_DIM = 256


#train = pd.read_csv('data/train.csv', sep=',', header=0)
test = pd.read_csv(args.test_path, sep=',', header=0)
movies = pd.read_csv(args.movie_path, sep='::', header=0)
users = pd.read_csv(args.user_path, sep='::', header=0)

#User_ID = train.values.T[1]
#Movie_ID = train.values.T[2]
#Rating = train.values.T[3]
#
#indices = np.arange(train.shape[0])
#np.random.shuffle(indices)
#User_ID = User_ID[indices]
#Movie_ID = Movie_ID[indices]
#Rating = Rating[indices]
#
#User_ID = np.array(User_ID).astype(int)
#Movie_ID = np.array(Movie_ID).astype(int)
#Rating = np.array(Rating).astype('float32')
#
normal = False
#if (normal):
#    print('normal')
#    Rating, mu, sigma = normalize(Rating)
#else:
#    print('no normal')
#    Rating = Rating - 3
#
##n_movies = 3883
##n_users = 6040
n_movies = 3952
n_users = 6040
#num = len(Movie_ID)
#
#User_ID =User_ID.reshape((num,1))
#Movie_ID =Movie_ID.reshape((num,1))
#Rating =Rating.reshape((num,1))

movie_input = keras.layers.Input(shape=[1])
movie_vec =keras.layers.Embedding(n_movies + 1, args.embedding_dim)(movie_input)
movie_vec =keras.layers.Flatten()(movie_vec)

user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Embedding(n_users + 1, args.embedding_dim)(user_input)
user_vec = keras.layers.Flatten()(user_vec)

Out = keras.layers.Dot(axes=1)([user_vec,movie_vec])
#Out = Dense(1)(Out)
#Out = Dropout(0.5)(Out)

movie_bias = keras.layers.Embedding(n_movies + 1, 1)(movie_input)
movie_bias = keras.layers.Flatten()(movie_bias)
#movie_bias = Dense(1)(movie_bias)
#movie_bias = Dropout(0.5)(movie_bias)

user_bias = keras.layers.Embedding(n_users + 1, 1)(user_input)
user_bias = keras.layers.Flatten()(user_bias)
#user_bias = Dense(1)(user_bias)
#user_bias = Dropout(0.5)(user_bias)

Out = keras.layers.Add()([Out,user_bias,movie_bias])


#Out = Dense(1)(Out)
#Out = Dropout(0.5)(Out)

#Out = Dense(1)(Out)
#Out = Dense(1)(Out)

model = keras.models.Model([user_input, movie_input], Out)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#		     metrics=['accuracy'])
print(model.summary())
#
load = True
if (load):
    model = load_model("hw5_1.h5")
    print(model.summary())
else:
    filepath="hw5_1.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
    earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')
    #
    callbacks_list = [checkpoint, earlystopping]
    model.fit([User_ID,Movie_ID],Rating,validation_split = 0.1, epochs=30, batch_size=2000,callbacks=callbacks_list)
    model = load_model("hw5_1.h5")

tes_1 = test.values.T[1]
tes_2 = test.values.T[2]

predict = model.predict([tes_1,tes_2], batch_size=128)
if (normal):
    mu = np.tile(mu, (predict.shape[0], 1))
    sigma = np.tile(sigma, (predict.shape[0], 1))
    predict = predict*sigma + mu
else:
    predict = predict + 3.0

with open(args.result_path, 'w') as f:
    f.write('TestDataID,Rating\n')
    for i, v in  enumerate(predict):
        f.write('%d,%f\n' %(i+1, v))
