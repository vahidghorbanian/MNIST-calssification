from utils import *
import pickle
import matplotlib.pyplot as plt


# %% Read and Load data
#train_img, train_lbl, train = load_dataset(dataset = "training", path = ".")
#test_img, test_lbl, test = load_dataset(dataset = "testing", path = ".")
pickle_in = open("data.pickle","rb")
example_dict = pickle.load(pickle_in)
train = example_dict['train']
test = example_dict['test']
train_img = example_dict['train_img']
test_img = example_dict['test_img']
train_lbl = example_dict['train_lbl']
test_lbl = example_dict['test_lbl']

print('Note that the number of training and test samples should be reduced to run the algorithm faster. ')
# %% classifications
#result_lr = lr_classifier(train, test)
#result_knn = knn_classifier(train, test)
#result_dt = dt_classifier(train, test)
#result_rf = rf_calssifier(train, test)
#result_Ada = Ada_calssifier(train, test)
#result_svm = SVM_classifier(train, test)
#result_nn = nn_classifier(train_img, train_lbl, test_img, test_lbl)
convNN_classifier(train_img, train_lbl, test_img, test_lbl)

#%% Plot NN results
#print('\nThe results are only plotted for the first trained NN model.')
#
## Plot losses vs epoch
#h = result_nn['history'][0].history
#epoch = np.arange(1, len(h['loss'])+1, 1)
#plt.figure()
#plt.plot(epoch, h['loss'])
#plt.plot(epoch, h['val_loss'])
#plt.legend(['train_loss', 'val_loss'])
#
## Plot some of misclassified samples
#predict_lbl = np.argmax(result_nn['prediction'][0],axis=1)
#test_lbl = result_nn['test_lbl']
#diff = np.nonzero(test_lbl-predict_lbl)[0]
#row = 7
#col = np.floor(len(diff)/row)
#plt.figure(figsize=(15,8))
#for i in np.arange(0,row*col,1):
#    plt.subplot(row, col, i+1)
#    plt.imshow(test_img[diff[int(i)]])
#    plt.tight_layout
#    plt.xlabel('as:'+str(predict_lbl[diff[int(i)]]))
#    plt.xticks([])
#    plt.yticks([])
#plt.suptitle('Misclassified Samples')
#plt.show()
    
