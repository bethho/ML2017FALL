import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np
import csv

from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend.tensorflow_backend as K
import tensorflow as tf
sys.path.insert(0, './')
import util
from util import DataManager
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.6

parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('model')
parser.add_argument('action', default = 'train', choices=['train','test','semi'])
parser.add_argument('--model_type', default='RNN')
parser.add_argument('--del_sign' , default = False)
parser.add_argument('--model_name', default = 'model.h5') 
# training argument
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--nb_epoch', default=20, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--gpu_fraction', default=0.3, type=float)
parser.add_argument('--vocab_size', default=20000, type=int) #65000
parser.add_argument('--max_length', default=40,type=int)

# model parameter
parser.add_argument('--loss_function', default='binary_crossentropy')
parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
parser.add_argument('-emb_dim', '--embedding_dim', default=128, type=int)
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
parser.add_argument('--threshold', default=0.1,type=float)

# output path for your prediction
parser.add_argument('--result_path', default='result.csv',)

# put model in the same directory
parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = 'model1/')
args = parser.parse_args()

train_path = 'data/training_label.txt'
test_path = 'data/testing_data.txt'
semi_path = 'data/training_nolabel.txt'

# build model
def simpleRNN(args):
    inputs = Input(shape=(args.max_length,))

    # Embedding layer
    embedding_inputs = Embedding(args.vocab_size,
                                 args.embedding_dim,
                                 trainable=True)(inputs)
    # RNN
    return_sequence = False
    dropout_rate = args.dropout_rate
    if args.cell == 'GRU':
        RNN_cell = GRU(args.hidden_size,
                       return_sequences=return_sequence,
                       dropout=dropout_rate)
    elif args.cell == 'LSTM':
        RNN_cell = LSTM(args.hidden_size,
                        return_sequences=return_sequence,
                        dropout=dropout_rate)

    RNN_output = RNN_cell(embedding_inputs)

    # DNN layer
    outputs = Dense(args.hidden_size // 2,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.1))(RNN_output)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    # optimizer
    adam = Adam()
    print('compile model...')

    # compile model
    model.compile(loss=args.loss_function, optimizer=adam, metrics=['accuracy', ])

    return model

def BOW(args):
    inputs = Input(shape=(args.vocab_size,))

    # DNN layer
    dropout_rate = args.dropout_rate

    outputs = Dense(args.hidden_size // 2,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.1))(inputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    # optimizer
    adam = Adam()
    print('compile model...')

    # compile model
    model.compile(loss=args.loss_function, optimizer=adam, metrics=['accuracy', ])

    return model

def main():
    # limit gpu memory usage
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(get_session(args.gpu_fraction))

    save_path = os.path.join(args.save_dir, args.model)
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir, args.load_model)

    #####read data#####
    dm = DataManager()
    print('Loading data...')
    if args.action == 'train':
        dm.add_data('train_data', train_path, True)
    elif args.action == 'semi':
        dm.add_data('train_data', train_path, True)
        dm.add_data('semi_data', semi_path, False)
    else:
        dm.add_testdata('test_data', test_path, False)

    # prepare tokenizer
    print('get Tokenizer...')
    print(args.del_sign)

    if args.load_model is not None:
        # read exist tokenizer
        dm.load_tokenizer(os.path.join(load_path, 'token.pk'))
    else:
        # create tokenizer on new data
        dm.tokenize(args.vocab_size, args.del_sign)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path, 'token.pk')):
        dm.save_tokenizer(os.path.join(save_path, 'token.pk'))

    # convert to sequences
    print(args.model_type)
    if args.model_type == 'RNN':
        dm.to_sequence(args.max_length)
    else:
        dm.to_bow()

    # initial model
    print ('initial model...')
    print(args.model_type)

    if args.model_type == 'RNN':
        model = simpleRNN(args)
    else:
        model = BOW(args)
    print (model.summary())

    save = save_path
    if args.load_model is not None:
        if args.action == 'train':
            print ('Warning : load a exist model and keep training')
        path = os.path.join(load_path, args.model_name)
        if os.path.exists(path):
            print ('load model from %s' % path)
            print(args.model_name)
            model.load_weights(path)
        else:
            raise ValueError("Can't find the file %s" %path)
    elif args.action == 'test':
        print ('Warning : testing without loading any model')

    # training
    if args.action == 'train':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model_{epoch:02d}-{val_acc:.2f}.h5')
        checkpoint = ModelCheckpoint(filepath=save_path,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_acc',
                                     mode='max' )
        train_history = model.fit(X, Y,
                            validation_data=(X_val, Y_val),
                            epochs=args.nb_epoch,
                            batch_size=args.batch_size,
                            verbose=2,
                            callbacks=[checkpoint, earlystopping] )

        A = train_history.history['acc']
        B = train_history.history['val_acc']
        AB = np.array([A,B]).T
        C = train_history.history['loss']
        D = train_history.history['val_loss']
        CD = np.array([C, D]).T

        if not os.path.isdir(save):
            os.makedirs(save)
        if not os.path.exists(os.path.join(save, 'acc.csv')):
            f = open(os.path.join(save, 'acc.csv'),'w')
            w = csv.writer(f)
            w.writerow(['acc', 'val_acc'])
            w.writerows(AB)
            f.close()
        if not os.path.exists(os.path.join(save, 'loss.csv')):
            f = open(os.path.join(save, 'loss.csv'),'w')
            w = csv.writer(f)
            w.writerow(['loss', 'val_loss'])
            w.writerows(CD)
            f.close()

    # testing
    elif args.action == 'test' :
        print("tsting data")
        X = dm.data['test_data'][0]
        prediction = model.predict(X)
        #Y = np.zeros((prediction.shape[0],1))
        #for i in range(prediction.shape[0]):
        #    Y[i] = float(prediction[i])
        labels = np.round(prediction)
        print('predict over')
        output_path = os.path.join(save, args.result_path)
        with open(output_path, 'w') as f:
            print(output_path)
            f.write('id,label\n')
            for i, v in enumerate(labels):
                f.write('%d,%d\n' % (i, v))

    # semi-supervised training
    elif args.action == 'semi':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)

        [semi_all_X] = dm.get_data('semi_data')
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max' )
        # repeat 10 times
        for i in range(10):
            # label the semi-data
            semi_pred = model.predict(semi_all_X, batch_size=1024, verbose=True)
            semi_X, semi_Y = dm.get_semi_data('semi_data', semi_pred, args.threshold, args.loss_function)
            semi_X = np.concatenate((semi_X, X))
            semi_Y = np.concatenate((semi_Y, Y))
            print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
            # train
            history = model.fit(semi_X, semi_Y,
                                validation_data=(X_val, Y_val),
                                epochs=2,
                                batch_size=args.batch_size,
                                callbacks=[checkpoint, earlystopping] )

            if os.path.exists(save_path):
                print ('load model from %s' % save_path)
                model.load_weights(save_path)
            else:
                raise ValueError("Can't find the file %s" % save_path)


if __name__ == '__main__':
        main()
