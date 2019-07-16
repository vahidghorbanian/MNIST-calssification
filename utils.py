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


num_test = 100
iter = 500
random_state = 0
def knn_classifier(train, test):
    print('\n******************************************')
    print('KNN is running...')
    print('The Euclidian distance (L2) is used.')
    num_k = [10]
    score_test = []
    for k in num_k:
        start_time = time.time()
        print('k = ', k)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train[:,:-1], train[:,-1])
        print('KNN fit done!')
        predict_test = model.predict(test[0:num_test,:-1])
        score_test.append(metrics.accuracy_score(test[0:num_test,-1], predict_test))
        elapsed_time = time.time() - start_time
        print('run time for k='+str(k)+': ', elapsed_time)
    print('Test scores: ', score_test)
    return {'model': model, 'test score': score_test}


def lr_classifier(train, test):
    print('\n******************************************')
    print('Logistic Regression is running...')
    C = [1]
    score_test = []
    for c in C:
        start_time = time.time()
        print('C = ', c)
        model = LogisticRegression(C=c, max_iter=iter, solver='lbfgs', multi_class='multinomial', random_state=random_state)
        print('Logisitic Regression fit done!')
        model.fit(train[:,:-1], train[:,-1])
        predict_test = model.predict(test[0:num_test,:-1])
        score_test.append(metrics.accuracy_score(test[0:num_test,-1], predict_test))
        elapsed_time = time.time() - start_time
        print('run time for C='+str(c)+': ', elapsed_time)
    print('Test scores: ', score_test)
    return {'model': model, 'test score': score_test}


def dt_classifier(train, test):
    print('\n******************************************')
    print('Decision Tree is running...')
    depth = [None]
    score_test = []
    for d in depth:
        start_time = time.time()
        print('depth = ', d)
        model = DecisionTreeClassifier(criterion='gini', max_depth=d, random_state=random_state)
        model.fit(train[:,:-1], train[:,-1])
        print('Decision Tree fit done!')
        predict_test = model.predict(test[0:num_test,:-1])
        score_test.append(metrics.accuracy_score(test[0:num_test,-1], predict_test))
        elapsed_time = time.time() - start_time
        print('run time for depth='+str(d)+': ', elapsed_time)
    print('Test scores: ', score_test)
    return {'model': model, 'test score': score_test}


def rf_calssifier(train, test):
    print('\n******************************************')
    print('Random forest is running...')
    depth = [None]
    score_test = []
    for d in depth:
        start_time = time.time()
        print('depth = ', d)
        model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=d, random_state=random_state)
        model.fit(train[:,:-1], train[:,-1])
        print('Random forest fit done!')
        predict_test = model.predict(test[0:num_test,:-1])
        score_test.append(metrics.accuracy_score(test[0:num_test,-1], predict_test))
        elapsed_time = time.time() - start_time
        print('run time for depth='+str(d)+': ', elapsed_time)
    print('Test scores: ', score_test)
    return {'model': model, 'test score': score_test, 'feature_importance': model.feature_importances_}


def Ada_calssifier(train, test):
    print('\n******************************************')
    print('AdaBoost is running...')
    estimator = [DecisionTreeClassifier(max_depth=2, random_state=random_state)]
    score_test = []
    for est in estimator:
        start_time = time.time()
        print('base estimator = ', est)
        model = AdaBoostClassifier(base_estimator=est, n_estimators=50, random_state=random_state)
        model.fit(train[:,:-1], train[:,-1])
        print('Ada Boost fit done!')
        predict_test = model.predict(test[0:num_test,:-1])
        score_test.append(metrics.accuracy_score(test[0:num_test,-1], predict_test))
        elapsed_time = time.time() - start_time
        print('run time: ', elapsed_time)
    print('Test scores: ', score_test)
    return {'model': model, 'test score': score_test, 'feature_importance': model.feature_importances_}





























