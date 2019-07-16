from utils import *
import pickle



# %% Read and Load data
#train_img, train_lbl, train = load_dataset(dataset = "training", path = ".")
#test_img, test_lbl, test = load_dataset(dataset = "testing", path = ".")
pickle_in = open("data.pickle","rb")
example_dict = pickle.load(pickle_in)
train = example_dict['train']
test = example_dict['test']


# %% classifications
result_knn = knn_classifier(train, test)
result_lr = lr_classifier(train, test)
result_dt = dt_classifier(train, test)
result_rf = rf_calssifier(train, test)
result_Ada = Ada_calssifier(train, test)
results_svm = SVM_classifier(train, test)
