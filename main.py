
# coding: utf-8

# In[1]:

import argparse
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
import keras
from keras.models import Sequential, Model, load_model
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize 
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
import sklearn
from keras import optimizers
import os
import math
import pandas as pd
print(keras.__version__)
print(sklearn.__version__)


# In[2]:

def train(opts):
    FOLDER = opts.name
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    MODEL_PATH = FOLDER + '/model.h5'
    FIG_PATH = FOLDER + '/history.png'
    #FIG_PATH_N = FOLDER + '/Confusion_Matrix_Norm.png'
    nclass = 39
    print(os.getpid())


    # In[3]:


    kdays = 120


    # In[4]:


    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Flatten
    from keras.layers import Conv2D, Conv1D, Convolution1D, Convolution2D
    from keras.layers import MaxPooling2D, MaxPooling1D, AveragePooling2D, GlobalMaxPooling1D, GlobalAveragePooling1D,BatchNormalization
    
    input_shape = (120, 39, 1)
    #input_shape = (5, 240)
    dprate = 0.8

    model = Sequential()
    model.add(Conv2D(72, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(72, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=None, padding='same'))
    model.add(Dropout(dprate))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=None, padding='same'))
    model.add(Dropout(dprate))


    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=None, padding='same'))
    #model.add(GlobalMaxPooling1D())
    #model.add(GlobalAveragePooling1D())

    model.add(Flatten())

    model.add(Dropout(dprate))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(dprate))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dprate))
    model.add(Dense(nclass))
    model.add(Activation('sigmoid'))



    model.summary()


    # In[5]:


    def imgshow(img):
        plt.imshow(np.reshape(img, (np.shape(img)[0], np.shape(img)[1])))
        plt.show()


    # In[6]:
    
    df = pd.read_csv(opts.source_data_path, index_col = 0)

    totaldays = df.shape[0]


    col = df.columns
    pics = []
    acts = []
    print(totaldays)


    # In[26]:


    for i in range(kdays, totaldays-1):
        pic = df[i-kdays:i].values
        pics.append(pic)

    label = df[kdays+1:].values



    label = np.asarray(label).astype('int')
    print(label)


    # In[32]:



    pics = np.asarray(pics)
    np.shape(pics[:])




    X = pics
    y = label

    X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], 1))
    print(np.shape(X))

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_test, y_train, y_test = X[:-240], X[-240:], y[:-240], y[-240:]

    print(np.shape(X_train))
    print(np.shape(y_train))

    #print(np.shape(X_train[0]))
    #imgshow(X_test[-1])
    #print(y_test[-1])
    #imgshow(X_train[-1])
    #print(y_train[-1])



    # In[7]:


    from keras.callbacks import LearningRateScheduler
     




    batch_size = opts.batch_size
    nb_epoch = 10000

    #class_weight = {0: 100.,1: 1.,2: 100.}
    earlystopping = EarlyStopping(monitor='val_loss', patience = opts.patience, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(filepath=MODEL_PATH, 
                                         verbose=1,
                                         save_best_only=True,
                                         monitor='val_loss',
                                         mode='auto', period = 10 )
    tb = TensorBoard(log_dir='./logs/' + FOLDER , write_graph=True)

    adam = optimizers.Adam(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
    history = model.fit(X_train, y_train, batch_size = batch_size, validation_data = (X_test, y_test), epochs = nb_epoch, verbose = 1,  shuffle = True, callbacks=[checkpoint, earlystopping, tb])


    # In[9]:


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'], loc = 'upper left')
    plt.savefig(FIGPATH)


    # In[10]:




    K = 3
    y_pred = model.predict(X_test)
    rank_pred = np.argsort(y_pred, axis = -1)


    # In[12]:


    results = np.zeros(np.shape(y_test)[0])

    for ind in range(np.shape(y_test)[0]):
        t = y_test[ind]
        p = rank_pred[ind]
        for i in range(1, K+1):
            if(t[p[-i]]==1):
                results[ind] += 1
               
    
    print(results)

    # In[13]:


    acc = results[results>=1].shape[0]/np.shape(results)[0]
    print("Top{} Acc = {:.2f}%".format(K, acc*100))


    # In[14]:


    RandShoot = 6/39 + 33/39 * 6/38 + 33/39*32/38*6/37
    print("Random Shoot for K=1 Acc: {:.2f}%".format(6/39*100))
    print("Random Shoot for K=3 Acc: {:.2f}%".format(RandShoot*100))


    # In[15]:


def test(opts):
    FOLDER = opts.name
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    MODEL_PATH = FOLDER + '/model.h5'
    
    model = load_model(MODEL_PATH)
    
    df = pd.read_csv(opts.source_data_path, index_col = 0)
    kdays = 120
    totaldays = df.shape[0]
    
    col = df.columns
    pics = []
    acts = []
    print(col)
    print(totaldays)
    
    for i in range(kdays, totaldays-1):
        pic = df[i-kdays:i].values
        pics.append(pic)
    
    X = np.asarray(pics)

    #testxpath = os.path.join(opts.source_data_path, 'X_test.npy')
    #X = np.load(testxpath).astype('float32')
    print(X.shape)
    x = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], 1))
    print(x.shape)
    y = model.predict(x)
    print(y)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--mode', type=str,
                        default='train', dest='mode',
                        help='train or test')
    
    parser.add_argument('--source_data_path', type=str,
                        default='./data', dest='source_data_path',
                        help='Path to source data')
    parser.add_argument('--name', type=str,
                        default='lottery', dest='name',
                        help='folder name to save and load')
    
    parser.add_argument('--batch_size', type=int,
                        default='120', dest='batch_size',
                        help='batch_size')
    parser.add_argument('--patience', type=int,
                        default='30', dest='patience',
                        help='patience')
    

    opts = parser.parse_args()
    
    if(opts.mode=='train'):
        train(opts)
    elif(opts.mode=='test'):
        test(opts)
        
