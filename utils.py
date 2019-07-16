import pandas as pd
import numpy as np
import struct
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import time


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


num_train = 20000
num_test = 500
iter = 500
random_state = 0
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
        model.append(LogisticRegression(C=c, max_iter=iter, solver='lbfgs', multi_class='multinomial', random_state=random_state))
        print('Logisitic Regression fit done!')
        model[i].fit(train[:,:-1], train[:,-1])
        predict_test = model[i].predict(test[0:num_test,:-1])
        score_test.append(metrics.accuracy_score(test[0:num_test,-1], predict_test))
        elapsed_time = time.time() - start_time
        print('run time: ', elapsed_time)
    print('Test scores: ', score_test)
    return {'model': model, 'test score': score_test}


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


def SVM_classifier(train, test):
    print('\n******************************************')
    print('SVM is running...')
    train = train[0:num_train,:]
    kernel = ['linear', 'poly', 'rbf']
#    kernel = ['rbf']
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



























