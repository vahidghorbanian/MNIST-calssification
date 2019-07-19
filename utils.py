import pandas as pd
import numpy as np
import struct
import os
import matplotlib.pyplot as plt
import h5py

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.nn import sigmoid, softmax 
import keras
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout
from keras.models import Model, Sequential
from keras import backend as K

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import time


#%%
def load_dataset(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 'test-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'test-labels.idx1-ubyte')        
# Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
# Extract train and test matrix
    mat = np.reshape(img[0],(1,28*28))
    for i, image in enumerate(img[1:]):
        mat = np.vstack((mat,np.reshape(image,(1,28*28))))
        print(dataset+' sample: ', i)
    mat = np.column_stack((mat, lbl))
    return img, lbl, mat


#%%
num_train = 60000
num_test = 10000
iter = 1000
random_state = 0
K.clear_session()

def lr_classifier(train, test):
    print('\n******************************************')
    print('Logistic Regression is running...')
    train = train[0:num_train,:]
    C = [1]
    score_test = []
    model = []
    for i, c in enumerate(C):
        start_time = time.time()
        print('\nC = ', c)
        model.append(LogisticRegression(C=c, max_iter=iter, solver='lbfgs', multi_class='auto', random_state=random_state))
        print('Logisitic Regression fit done!')
        model[i].fit(train[:,:-1], train[:,-1])
        predict_test = model[i].predict(test[0:num_test,:-1])
        score_test.append(metrics.accuracy_score(test[0:num_test,-1], predict_test))
        elapsed_time = time.time() - start_time
        print('run time: ', elapsed_time)
    print('Test scores: ', score_test)
    return {'model': model, 'test score': score_test}


#%%
def knn_classifier(train, test):
    print('\n******************************************')
    print('KNN is running...')
    print('The Euclidian distance (L2) is used.')
    train = train[0:num_train,:]
    num_k = [3, 10]
    score_test = []
    model = []
    for i, k in enumerate(num_k):
        start_time = time.time()
        print('\nk = ', k)
        model.append(KNeighborsClassifier(n_neighbors=k))
        model[i].fit(train[:,:-1], train[:,-1])
        print('KNN fit done!')
        predict_test = model[i].predict(test[0:num_test,:-1])
        score_test.append(metrics.accuracy_score(test[0:num_test,-1], predict_test))
        elapsed_time = time.time() - start_time
        print('run time: ', elapsed_time)
    print('Test scores: ', score_test)
    return {'model': model, 'test score': score_test}


#%%
def dt_classifier(train, test):
    print('\n******************************************')
    print('Decision Tree is running...')
    train = train[0:num_train,:]
    depth = [None]
    score_test = []
    model = []
    for i, d in enumerate(depth):
        start_time = time.time()
        print('\ndepth = ', d)
        model.append(DecisionTreeClassifier(criterion='gini', max_depth=d, random_state=random_state))
        model[i].fit(train[:,:-1], train[:,-1])
        print('Decision Tree fit done!')
        predict_test = model[i].predict(test[0:num_test,:-1])
        score_test.append(metrics.accuracy_score(test[0:num_test,-1], predict_test))
        elapsed_time = time.time() - start_time
        print('run time: ', elapsed_time)
    print('Test scores: ', score_test)
    return {'model': model, 'test score': score_test}


#%%
def rf_calssifier(train, test):
    print('\n******************************************')
    print('Random forest is running...')
    train = train[0:num_train,:]
    depth = [None]
    score_test = []
    model = []
    for i, d in enumerate(depth):
        start_time = time.time()
        print('\ndepth = ', d)
        model.append(RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=d, random_state=random_state))
        model[i].fit(train[:,:-1], train[:,-1])
        print('Random forest fit done!')
        predict_test = model[i].predict(test[0:num_test,:-1])
        score_test.append(metrics.accuracy_score(test[0:num_test,-1], predict_test))
        elapsed_time = time.time() - start_time
        print('run time for depth='+str(d)+': ', elapsed_time)
    print('Test scores: ', score_test)
    return {'model': model, 'test score': score_test}


#%%
def Ada_calssifier(train, test):
    print('\n******************************************')
    print('AdaBoost is running...')
    train = train[0:num_train,:]
    estimator = [DecisionTreeClassifier(max_depth=2, random_state=random_state)]
    score_test = []
    model= []
    for i, est in enumerate(estimator):
        start_time = time.time()
        print('\nbase estimator = ', est)
        model.append(AdaBoostClassifier(base_estimator=est, n_estimators=50, random_state=random_state))
        model[i].fit(train[:,:-1], train[:,-1])
        print('Ada Boost fit done!')
        predict_test = model[i].predict(test[0:num_test,:-1])
        score_test.append(metrics.accuracy_score(test[0:num_test,-1], predict_test))
        elapsed_time = time.time() - start_time
        print('run time: ', elapsed_time)
    print('Test scores: ', score_test)
    return {'model': model, 'test score': score_test}


#%%
def SVM_classifier(train, test):
    print('\n******************************************')
    print('SVM is running...')
    train = train[0:num_train,:]
    kernel = ['linear', 'poly', 'rbf']
    C = [1, 10, 100, 1000, 10000]
    score_test = []
    score = []
    model = []
    count = 0
    for i, k in enumerate(kernel):
        for j, c in enumerate(C):
            start_time = time.time()
            print('\nkernel =', k, ', C =', c)
            model.append(SVC(C=1000, kernel=k, degree=9, gamma ='scale', random_state=random_state))
            model[count].fit(train[:,:-1], train[:,-1])
            print('SVC fit done!')
            predict_test = model[count].predict(test[0:num_test,:-1])
            score.append(metrics.accuracy_score(test[0:num_test,-1], predict_test))
            elapsed_time = time.time() - start_time
            print('run time: ', elapsed_time)
            count = count + 1
        score_test.append(score)
    print('Test scores: ', score_test)
    return {'model': model, 'test score': score_test}


#%%
def nn_classifier(train_img, train_lbl, test_img, test_lbl):
    print('\n******************************************')
    print('fully connected neural net is running...')
#    train = train[0:num_train,:]
    train_img = train_img[0:num_train,:]
    test_img = test_img[0:num_test,:]
    train_lbl = train_lbl[0:num_train]
    test_lbl = test_lbl[0:num_test]
    # convert class vectors to binary class matrices
    num_classes = 10
    train_lbl = keras.utils.to_categorical(train_lbl, num_classes)
    test_lbl = keras.utils.to_categorical(test_lbl, num_classes)
    
    num_h = 64
    epochs = 200
    l1_val = 0.0
    l2_val = 0.0
    earlystop = True
    reg = regularizers.l1_l2(l1=l1_val, l2=l2_val)
    model = Sequential()
    model.add(Flatten(input_shape=(np.shape(train_img)[1], np.shape(train_img)[2])))
    model.add(Dense(num_h, activation=sigmoid, kernel_regularizer=reg))
    model.add(Dense(num_h, activation=sigmoid, kernel_regularizer=reg))
    model.add(Dense(10, activation=tf.nn.softmax)) 

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if earlystop == False:
        print('Early Stopping deactivated!')
        hist.append(m.fit(train_img, train_lbl, epochs=epochs))
    else:
        print('Early Stopping activated!')
        callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
        hist = model.fit(train_img, train_lbl, epochs=epochs, callbacks=callbacks,
                     validation_split =0.2, shuffle=True)
    print('\ncalculate test score')
    score_test = model.evaluate(test_img, test_lbl)
    predict = model.predict(test_img)
    w = model.get_weights()
    print('number of hidden units:', num_h)
    print('test scores:\n', list(score_test))
    return {'model': model, 'num_hidden_units': num_h, 'test_score': score_test,
            'prediction': predict, 'test_lbl': test_lbl, 'test_img': test_img,
            'weights': w, 'history': hist}
    
    
# %%
def convNN_classifier(train_img, train_lbl, test_img, test_lbl):
    print('\n******************************************')
    print('Convolutional neural net is running...')
    # train = train[0:num_train,:]
    train_img = train_img[0:num_train,:]
    test_img = test_img[0:num_test,:]
    train_lbl = train_lbl[0:num_train]
    test_lbl = test_lbl[0:num_test]
    
    # Reshape for channel_last
    train_img = train_img.reshape(train_img.shape[0], 28, 28, 1)
    test_img = test_img.reshape(test_img.shape[0], 28, 28, 1)
    
    # cobvert to float32 and normalize to 1
    train_img = train_img.astype('float32')
    test_img = test_img.astype('float32')
    train_img /= 255
    test_img /= 255
    input_shape = (28, 28, 1)
    
    # convert class vectors to binary class matrices
    num_classes = 10
    train_lbl = keras.utils.to_categorical(train_lbl, num_classes)
    test_lbl = keras.utils.to_categorical(test_lbl, num_classes)
    
    # Build Conv2dNN model
    batch_size = 128
    epochs = 12
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])
    
    checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

    hist = model.fit(train_img, train_lbl,
                     batch_size=batch_size, 
                     epochs=epochs,
                     verbose=1,
                     callbacks=[checkpointer],
                     validation_split=0.2,
                     validation_data=None)
    
    score_test = model.evaluate(test_img, test_lbl, verbose=0)
    predict = model.predict(test_img)
    print(score_test)
    model.load_weights('best_weights.hdf5')
    weights = model.get_weights()
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    return {'model': model, 'test_score': score_test, 'prediction': predict,
            'test_lbl': test_lbl, 'test_img': test_img, 'history': hist,
            'activation_model': activation_model}
    





















