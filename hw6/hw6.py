# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import sys, argparse, os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.6

parser = argparse.ArgumentParser(description='encoder')
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--image_file', type=str)
parser.add_argument('--test_file', type=str)
parser.add_argument('--result', type=str)
args = parser.parse_args()
#print(args.embedding_dim)

# load data
#img = np.load('data/image.npy')
img = np.load(args.image_file)
img = img.astype('float32') / 255.
#encoding_dim = 32
encoding_dim = args.embedding_dim

# build model
#input_img = Input(shape=(784,))
## encoder layers
#encoded = Dense(512, activation='relu')(input_img)
##encoded = Dense(256, activation='relu')(encoded)
#encoded = Dense(256, activation='relu')(encoded)
#encoder_output = Dense(encoding_dim)(encoded)
#
## decoder layers
#decoded = Dense(256, activation='relu')(encoder_output)
#decoded = Dense(512, activation='relu')(decoded)
##decoded = Dense(128, activation='relu')(decoded)
#decoded = Dense(784, activation='tanh')(decoded)
#
## construct the autoencoder model
#autoencoder = Model(input=input_img, output=decoded)
## compile autoencoder
#autoencoder.compile(optimizer='adam', loss='mse')
#
#train = False
#if train == True:
#    # training
#    autoencoder.fit(img, img,
#                    nb_epoch=105,
#                    batch_size=256,
#                    shuffle=True)
#    # construct the encoder model for plotting
#    encoder = Model(input=input_img, output=encoder_output)
#    encoder.save('result/model/encoder.h5')
#else:
encoder = load_model('encoder.h5')
#encoder = load_model('hw6_112_great.h5')
code = encoder.predict(img)


#data_path = 'data/test_case.csv'
data_path = args.test_file
test_data = pd.read_csv(data_path, sep=',',encoding='utf-8')
test = test_data.values
result = np.zeros((len(test),))

cluster = KMeans(n_clusters=2).fit(code)
class_ = cluster.labels_
for i, id_ in enumerate(test):
    temp1 = class_[id_[1]]
    temp2 = class_[id_[2]]
    result[i] = (temp1==temp2).astype(int)

#result_path = 'result/result_encoder.csv'
result_path = args.result
with open(result_path, 'w') as f:
    f.write('ID,Ans\n')
    for i, v in  enumerate(result):
        f.write('%d,%d\n' %(i, v))