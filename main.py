from utils import *
import pickle

# %% Read and Load data
# Uncomment either the following two lines OR the next 8 lines in order to read and load data 

# METHOD 1
#train_img, train_lbl, train = load_dataset(dataset = "training", path = ".")
#test_img, test_lbl, test    = load_dataset(dataset = "testing", path = ".")

# METHOD 2
pickle_in = open("data.pickle","rb")
example_dict = pickle.load(pickle_in)
train = example_dict['train']
test = example_dict['test']
train_img = example_dict['train_img']
test_img = example_dict['test_img']
train_lbl = example_dict['train_lbl']
test_lbl = example_dict['test_lbl']

print('Note that the number of training and test samples should be reduced to run the algorithm faster.')
print('This can be done in the Initialization section of utils.py')
print('\nNumber of training samples = ', num_train)
print('Number of test samples = ', num_test)

# %% classifications
result_lr  = lr_classifier(train, test)
result_knn = knn_classifier(train, test) 
result_dt  = dt_classifier(train, test)
result_rf  = rf_calssifier(train, test)
result_Ada = Ada_calssifier(train, test)
result_svm = SVM_classifier(train, test)
result_nn  = nn_classifier(train_img, train_lbl, test_img, test_lbl)
result_cnn = convNN_classifier(train_img, train_lbl, test_img, test_lbl)

#%% Visualize CNN results
print('\nVisualize CNN layersoutput for one test sample')
model = result_cnn['model']
model.summary()
activation_model = result_cnn['activation_model']
test_img = result_cnn['test_img']
im = test_img[23].reshape(1, test_img.shape[1], test_img.shape[2], 1)
activations = activation_model.predict(im)
for idx, act in enumerate(activations[0:4]):
    row = 4
    col = act.shape[-1] / row
    plt.figure(figsize=(12, 6))
    for i in np.arange(0, act.shape[-1]):       
        plt.subplot(row, col, i+1)
        plt.imshow(act[0,:,:,i], cmap='viridis')
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('Layer index: '+str(idx+1)+'\nLayer type: '+model.layers[idx].get_config()['name'])
plt.show()
    
