### Import packages and Setup

# Mount Google Drive
#from google.colab import drive
#drive.mount('/gdrive')

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(y_true, y_pred, title):
  # calculate scores
  auc = roc_auc_score(y_true, y_pred)
  # summarize scores
  print('Area Under Curve=%.3f' % (auc))
  # calculate roc curves
  fpr, tpr, _ = roc_curve(y_true, y_pred)
  # plot the roc curve for the model
  plt.plot(fpr, tpr, marker='.', label='ROC')
  # axis labels
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(title)
  # show the legend
  plt.legend()
  # show the plot
  plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix

def score_model(model, x, y):
    #scores = model.evaluate(x, y, verbose=0)  
    y_pred = model.predict(x)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    conf_matrix = confusion_matrix(y_true=y, y_pred=y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    sensitivity = tp / (tp+fn)
    specificity = tn / (fp+tn)

    

    print("Accuracy: %.2f Sensitivity: %.2f Specificity: %.2f" % (accuracy, sensitivity, specificity))
    return np.reshape( (accuracy, sensitivity,specificity) , (1,3) )  

# assumes minority class has label 1
def SMOTE(x_train, y_train):
  minority_num = np.sum(y_train)
  majority_samples=[]
  minority_samples=[]
  labels =[]
  for i in range (len(y_train)):
    if(y_train.iloc[i]==0):
      majority_samples.append(x_train.iloc[i])
    else:
      minority_samples.append(x_train.iloc[i])
      labels.append(1)

  samples = minority_samples

  majority_num = np.shape(majority_samples)[0]
  for i in range(minority_num):
    index = int(majority_num * np.random.rand())
    samples.append(majority_samples[index])
    labels.append(0)

  return samples, labels

### Read Data and Pre-Processing

###DOMINIK PATH
# Define Location of Project Folder
drive_path = '/gdrive/My Drive/NTU/FinTech/Fin Tech: Final presentation/Final Project/'
# Define Locations of Datasets
german_data = 'Datasets/german_credit_data/german_credit_data.csv'
taiwan_data = 'Datasets/UCI_Credit_Card/UCI_Credit_Card.csv'

# Read and print german data
german_raw = read_csv(drive_path+german_data)
print(german_raw)

# Read and print UCI data
taiwan_raw = read_csv(drive_path+taiwan_data)
print(taiwan_raw)

# Fetch Label 
taiwan_y = taiwan_raw['default.payment.next.month']
# Remove ID and Label from Dataset
taiwan_raw.drop(columns='default.payment.next.month', inplace=True)
taiwan_raw.drop(columns='ID', inplace=True)

# Dataset Distribution
print(taiwan_y.value_counts())
plt.bar(["Non default","Default"],taiwan_y.value_counts(), color=['royalblue','darkorange'])
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title("Original dataset distribution")

# No Normalization
#taiwan_x = taiwan_raw

# Remove Mean and Standard Deviation
#mean = taiwan_raw.mean(axis=0)
#std= taiwan_raw.std(axis=0)
#taiwan_x = (taiwan_raw - mean) / std
#print(taiwan_x.mean(axis=0))
#print(taiwan_x.std(axis=0))

taiwan_x_train, taiwan_x_temp, taiwan_y_train, taiwan_y_temp = train_test_split(taiwan_raw, taiwan_y, test_size=0.2)
taiwan_x_val, taiwan_x_test, taiwan_y_val, taiwan_y_test = train_test_split(taiwan_x_temp, taiwan_y_temp, test_size=0.1/0.2)

# Rescale Data between 0...1
taiwan_min = np.min(taiwan_x_train)
taiwan_max = np.max(taiwan_x_train)
taiwan_x_train = (taiwan_x_train-taiwan_min) / (taiwan_max-taiwan_min)
taiwan_x_val = (taiwan_x_val-taiwan_min) / (taiwan_max-taiwan_min)
taiwan_x_test = (taiwan_x_test-taiwan_min) / (taiwan_max-taiwan_min)

print(np.min(taiwan_x_train), " ", np.max(taiwan_x_train))
print(np.min(taiwan_x_val), " ", np.max(taiwan_x_val))
print(np.min(taiwan_x_test), " ", np.max(taiwan_x_test))

#taiwan_x_train, taiwan_y_train = SMOTE(taiwan_x_train, taiwan_y_train)

# Split Dataset in 80% Training and 20% Test
#taiwan_x_train, taiwan_x_test, taiwan_y_train, taiwan_y_test = train_test_split(taiwan_x, taiwan_y, test_size=0.2)
print(np.shape(taiwan_x_train))
print(np.shape(taiwan_x_val))
print(np.shape(taiwan_x_test))
print(np.shape(taiwan_y_train), " ", np.sum(taiwan_y_train), " Defaults")
print(np.shape(taiwan_y_val), " ", np.sum(taiwan_y_val), " Defaults")
print(np.shape(taiwan_y_test), " ", np.sum(taiwan_y_test), " Defaults")

### Model 1 - k-nearest neighbors

knn_scores = np.ndarray((0,4))

for k in range (1,101,2):
  print(k)
  # Create new Classifier
  KNN = KNeighborsRegressor(n_neighbors=k)
  # Fit training Data to Classifier
  KNN.fit(taiwan_x_train, taiwan_y_train)
  # Evaluate Classifier Performance on Validation Set
  scores = score_model(KNN, taiwan_x_val, taiwan_y_val)
  knn_scores = np.append(knn_scores, np.append(np.reshape(k, (1,1)), scores, axis=1), axis=0)

df=pd.DataFrame(knn_scores)
df.to_csv(drive_path+'Evaluation/eval_knn_best_k.csv', header=False, sep=';', decimal=',')

best_acc  = knn_scores[int(np.argmax(knn_scores[:,[1]], axis=0))]
best_sens = knn_scores[int(np.argmax(knn_scores[:,[2]], axis=0))]
best_spec = knn_scores[int(np.argmax(knn_scores[:,[3]], axis=0))]
print(best_acc)
print(best_sens)
print(best_spec)
best_acc_k  = best_acc[0]
best_sens_k = best_sens[0]
best_spec_k = best_spec[0]
print("Maximum Accuracy: "    , best_acc_k)
print("Maximum Sensitivity: " , best_sens_k)
print("Maximum Specificity: " , best_spec_k)
#print(np.max(knn_scores, axis=0)[1:])

plt.figure()
plt.plot(knn_scores[:,[0]], knn_scores[:,[1]], label='Accuracy')
plt.plot(knn_scores[:,[0]], knn_scores[:,[2]], label='Sensitivity')
plt.plot(knn_scores[:,[0]], knn_scores[:,[3]], label = 'Specificity')
plt.title('K-Nearest-Neighbors Hyperparameter k')
plt.xlabel('Hyperparamter k')
plt.ylabel('Performance')
plt.legend()
plt.show()

optimal_k=11

train_size_scores = np.ndarray((0,4))


steps = 100

for i in range (1,steps+1):  
  taiwan_x_cut = taiwan_x_train[:int(i/steps*len(taiwan_x_train))]
  taiwan_y_cut = taiwan_y_train[:int(i/steps*len(taiwan_y_train))]  

  #print(np.shape(taiwan_x_cut))
  #print(np.shape(taiwan_y_cut))

  # Create new Classifier
  KNN = KNeighborsRegressor(n_neighbors=optimal_k)
  # Fit training Data to Classifier
  KNN.fit(taiwan_x_cut, taiwan_y_cut)
  # Evaluate Classifier Performance on Validation Set
  scores = score_model(KNN, taiwan_x_val, taiwan_y_val)
  train_size_scores = np.append(train_size_scores, np.append(np.reshape(i*100/steps, (1,1)), scores, axis=1), axis=0)

df=pd.DataFrame(train_size_scores)
df.to_csv(drive_path+'Evaluation/eval_knn_data_size.csv', header=False, sep=';', decimal=',')
  


#print(lift_scores[:,[0]])
plt.figure()
plt.plot(train_size_scores[:,[0]], train_size_scores[:,[1]])
plt.title('K-Nearest-Neighbors Performance / Training Data')
plt.xlabel('Dataset Size / %')
plt.ylabel('Accuracy')
plt.show()

# Final Prediction of Test Dataset
test_scores = np.ndarray((0,4))
# Create new Classifier
KNN = KNeighborsRegressor(n_neighbors=optimal_k)
# Fit training Data to Classifier
KNN.fit(taiwan_x_train, taiwan_y_train)
# Evaluate Classifier Performance on Test Set
scores = score_model(KNN, taiwan_x_test, taiwan_y_test)
test_scores = np.append(test_scores, np.append(np.reshape(best_acc_k, (1,1)), scores, axis=1), axis=0)

df=pd.DataFrame(test_scores)
df.to_csv(drive_path+'Evaluation/eval_knn_predict_test.csv', header=False, sep=';', decimal=',')

y_pred = KNN.predict(taiwan_x_test)
plot_roc_curve(taiwan_y_test, y_pred, 'K-Nearest-Neighbors ROC-Curve')

!pip install scikit-plot

import scikitplot as skplt
import matplotlib.pyplot as plt
# plot functions need "one hot encoded" class probabilities
y_neg = 1 - y_pred
y_both = np.append(np.reshape(y_neg, (-1,1)), np.reshape(y_pred, (-1,1)), axis=1)
# plot curves
skplt.metrics.plot_cumulative_gain(taiwan_y_test, y_both)
plt.show()
skplt.metrics.plot_lift_curve(taiwan_y_test, y_both)
plt.show()

skplt.metrics.plot_confusion_matrix(taiwan_y_test, np.argmax(y_both, axis=1), normalize=True)
plt.show()



```
# Ce texte est au format code
```

### Model 2 - Decision Tree

# =============================================================================
# SMOTE + DOWNSAMPLING
# =============================================================================

# assumes minority class has label 1
def SMOTE(x_train, y_train, show_repartition = False):
    minority_samples=[]
    for i in range (len(y_train)):
        if(y_train.iloc[i]==1):
            minority_samples.append(x_train.iloc[i])
            
    x_temp = []
    y_temp = y_train.values
    # We add as many sample with label=1 as needed
    for i in range(len(x_train)-2*len(minority_samples)):
        index1 = int(len(minority_samples) * np.random.rand())
        index2 = int(len(minority_samples) * np.random.rand())
        
        # Add a point between 2 random points
        x_temp.append((x_train.iloc[index1]+x_train.iloc[index2])/2)
        y_temp = np.append(y_temp, [1], axis=0)
    
    if show_repartition:
        plt.bar([0,1],[x_train.shape[0]-len(minority_samples), len(minority_samples)])
        plt.xticks([0,1])
        plt.xlabel('0 : normal | 1 : default')
        plt.ylabel('number of sample')
        plt.title('Unbalanced data set', fontsize=18, weight = 'bold')
    
    return x_train.append(pd.DataFrame(x_temp)), y_temp

def downsample(trainX, trainY):

  print('Proportion of positive labels to equalize : ', np.sum(trainY)/len(trainY))
  eject_prob = 1-np.sum(trainY)/len(trainY)
  
  X = np.array(trainX)
  Y = np.array(trainY)
  indexes_to_delete = []

  for i in range(len(trainX)):
    if Y[i] ==0: #this is the larger class'label : we want to reduce the individuals number in that class
      p = np.random.uniform()
      if p <= eject_prob:
        indexes_to_delete.append(i)

  X = np.delete(X, indexes_to_delete, axis = 0)
  Y = np.delete(Y, indexes_to_delete, axis = 0)
  print(len(indexes_to_delete), ' values deleted')
  print('New label balance : ', np.sum(Y)/len(Y))
  print('Size reduction : Original ', len(trainY), ' -->  ',len(Y))

  return X, Y

**Test hyperparameters**

from sklearn import tree

def create_tree(X, Y, Xtest, Ytest, n):
    """create n trees with a depth from 1 to n"""
    results = np.zeros((n,3))
    for i in range(1, n):
        # create and plot the tree 
        clf = tree.DecisionTreeClassifier(min_samples_leaf= 10, max_depth= i)
        clf = clf.fit(X, Y)
    
        #print the accuracy of the model
        results[i-1,:] = score_model(clf, Xtest, Ytest)
    return results[:n-1,:]

def plot_tree(metrics):
    """plot the result curves with an array input [Accuracy, Sensitivity, Specificity]"""
    #Print them on 2 graphs
    n = np.arange(1,metrics.shape[0]+1)
    print(n)
    plt.plot(n, metrics[:,0], label="Accuracy")
    plt.legend(loc='lower center')
    plt.xlabel('n : tree depth')
    plt.ylabel('ACCURACY')

    plt.figure()
    plt.plot(n, metrics[:,1], label="Sensivity")
    plt.plot(n, metrics[:,2], label="Specificity")
    plt.legend( loc='lower right', ncol=1)
    plt.xlabel('n : tree dept')
    plt.ylabel('metrics')
    plt.show()
   
    plt.figure()
    plt.plot(n, metrics[:,0], label="Accuracy")
    plt.plot(n, metrics[:,1], label="Sensivity")
    plt.plot(n, metrics[:,2], label="Specificity")
    plt.legend( loc='lower right', ncol=1)
    plt.xlabel('n : tree dept')
    plt.ylabel('metrics')
    plt.show()

### test with several parameters
results1 = create_tree( taiwan_x_train, taiwan_y_train, taiwan_x_val, taiwan_y_val, 20)
# plot_tree(results1)


### test with the choosen parameter
clf = tree.DecisionTreeClassifier(min_samples_leaf= 10, max_depth= 4)
clf = clf.fit(taiwan_x_train, taiwan_y_train)
score_model(clf, taiwan_x_val, taiwan_y_val)

y_pred = clf.predict(taiwan_x_test)

**Test the stability of the prediction (abscice = different tries)**

def test_stability(parameter ="Vanilla"):
    results = np.zeros((8,3))
    for i in range(8):
        taiwan_x_train, taiwan_x_temp, taiwan_y_train, taiwan_y_temp = train_test_split(taiwan_raw, taiwan_y, test_size=0.2)
        taiwan_x_val, taiwan_x_test, taiwan_y_val, taiwan_y_test = train_test_split(taiwan_x_temp, taiwan_y_temp, test_size=0.1/0.2)
        
        # Rescale Data between 0...1
        taiwan_min = np.min(taiwan_x_train)
        taiwan_max = np.max(taiwan_x_train)
        taiwan_x_train = (taiwan_x_train-taiwan_min) / (taiwan_max-taiwan_min)
        taiwan_x_val = (taiwan_x_val-taiwan_min) / (taiwan_max-taiwan_min)
        taiwan_x_test = (taiwan_x_test-taiwan_min) / (taiwan_max-taiwan_min)

        # Apply the transformation
        if parameter == "SMOTE":
          taiwan_x_train, taiwan_y_train = SMOTE(taiwan_x_train, taiwan_y_train)
        elif parameter == "downsample":
          taiwan_x_train, taiwan_y_train = downsample(taiwan_x_train, taiwan_y_train)

        clf = tree.DecisionTreeClassifier(min_samples_leaf= 10, max_depth= 5)
        clf = clf.fit(taiwan_x_train, taiwan_y_train)
        results[i-1,:] = score_model(clf, taiwan_x_val, taiwan_y_val)
    n = np.arange(1,results.shape[0]+1)
    plt.plot(n, results[:,0], label="Accuracy")
    plt.plot(n, results[:,1], label="Sensivity")
    plt.plot(n, results[:,2], label="Specificity")
    plt.legend( loc='lower right', ncol=1)
    plt.xlabel('different data split')
    plt.ylabel('metrics')
    plt.ylim((0.3,1))
    plt.show()
    
test_stability()
test_stability(parameter = "SMOTE")
test_stability(parameter = "downsample")

### Export the tree on a pdf
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None, max_depth= 5,
                filled=True, rounded=True,  
                special_characters=True)  
graph = graphviz.Source(dot_data) 
graph.render("Taiwanese_descision_tree2") 

def test_amount_data(steps = 10):
    """ give result whith different amout of data"""
    train_size_scores = np.ndarray((0,4))
    for i in range (1,steps+1):  
      taiwan_x_cut = taiwan_x_train[:int(i/steps*len(taiwan_x_train))]
      taiwan_y_cut = taiwan_y_train[:int(i/steps*len(taiwan_y_train))]  
    
      #print(np.shape(taiwan_x_cut))
      #print(np.shape(taiwan_y_cut))
    
      # Create new Classifier
      clf = tree.DecisionTreeClassifier(min_samples_leaf= 10, max_depth= 5)
      # Fit training Data to Classifier
      clf = clf.fit(taiwan_x_cut, taiwan_y_cut)
      # Evaluate Classifier Performance on Validation Set
      scores = score_model(clf, taiwan_x_val, taiwan_y_val)
      train_size_scores = np.append(train_size_scores, np.append(np.reshape(i*100/steps, (1,1)), scores, axis=1), axis=0)
    #print(lift_scores[:,[0]])
    plt.figure()
    plt.plot(train_size_scores[:,[0]], train_size_scores[:,[1]])
    plt.title('Descision tree Performance / Training Data')
    plt.xlabel('Dataset Size / %')
    plt.ylabel('Accuracy')
    plt.show()
    
test_amount_data()

!pip install scikit-plot

**Lift curves**

import scikitplot as skplt

# plot functions need "one hot encoded" class probabilities
y_neg = 1 - y_pred
y_both = np.append(np.reshape(y_neg, (-1,1)), np.reshape(y_pred, (-1,1)), axis=1)
# plot curves
skplt.metrics.plot_cumulative_gain(taiwan_y_test, y_both)
plt.show()
skplt.metrics.plot_lift_curve(taiwan_y_test, y_both)
plt.show()

**Confusion matrix**

skplt.metrics.plot_confusion_matrix(taiwan_y_test, np.argmax(y_both, axis=1), normalize=True)
plt.show()

# Random forrest for the comparison


from sklearn.ensemble import RandomForestClassifier

def create_forrest(X ,Y, Xtest, Ytest, n):
    """create n random forrest trees from 1 to n"""
    results = np.zeros((n,3))
    for i in range(1,n):
        rf = RandomForestClassifier(n_estimators=i)
        rf = rf.fit(X, Y)
        results[i-1,:] = score_model(rf, Xtest, Ytest)
    return results[:n-1,:]

def plot_forest(metrics):
    """plot the result curves with an array input [Accuracy, Sensitivity, Specificity]"""
    #Print them on 2 graphs
    n = np.arange(1,metrics.shape[0]+1)
    # print(n)
    plt.plot(n, metrics[:,0], label="Accuracy")
    plt.legend(loc='lower center')
    plt.xlabel('n : tree number')
    plt.ylabel('ACCURACY')

    plt.figure()
    plt.plot(n, metrics[:,1], label="Sensivity")
    plt.plot(n, metrics[:,2], label="Specificity")
    plt.legend( loc='lower right', ncol=1)
    plt.xlabel('n : tree number')
    plt.ylabel('metrics')
    plt.show()
   
##print the results
results2 = create_forrest( taiwan_x_train, taiwan_y_train, taiwan_x_val, taiwan_y_val, 30)
plot_forest(results2)   


### Model 3 - SVM

from sklearn.svm import SVC, LinearSVC
import timeit
start = timeit.default_timer()

# Regular SVM (RBF) - Hyperparameters for high accuracy but average recall
# Note : Using SMOTE has the same performance as using class weights balancing which is  already implemented in SVM function.
#        In both cases, balancing the dataset improves recall / specificity but reduces the accuracy
print("#### SVM - RBF Kernel ####")
SVM = SVC(C = 10, gamma=1, kernel='rbf' , class_weight="balanced")
SVM.fit(taiwan_x_train, taiwan_y_train)
scores = score_model(SVM, taiwan_x_test, taiwan_y_test)
print(scores)

# Linear SVM - Hyperparameters for a high recall but average accuracy
# Note : LinearSVM scales better with more samples, the training is faster than regular RBF SVM.
print("#### SVM - Linear - 32K iterations ####")
LinearSVM = LinearSVC(C = 0.001,penalty='l1',dual=False, max_iter=32000, class_weight="balanced")
LinearSVM.fit(taiwan_x_train, taiwan_y_train)
scores = score_model(LinearSVM, taiwan_x_test, taiwan_y_test)
print(scores)



Grid Search - Hyperparameters tuning

# # GridSearch 
# from sklearn.svm import SVC, LinearSVC
# from sklearn.model_selection import GridSearchCV
# import timeit


# def svc_param_selection(X, y, nfolds):
#     Cs = [0.001, 0.01, 0.1, 1, 10]
#     gammas = [0.001, 0.01, 0.1, 1]
#     kernels = ['linear', 'rbf', 'poly'] # 20min with only linear kernel ...
#     param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernels}
#     grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
#     grid_result = grid_search.fit(X, y)


#     # GridSearchCV object store training results in attribute :  cv_results_
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     return grid_search.best_params_

# # Concatenate x_train and x_val for GridSearch (Cross validation)
# taiwan_x_gridsearch = pd.concat([taiwan_x_train, taiwan_x_val], ignore_index=True)
# taiwan_y_gridsearch = pd.concat([taiwan_y_train, taiwan_y_val], ignore_index=True)

# start = timeit.default_timer()
# # Use all the dataset (cross-validation in GridSearchCV function)
# best_params = svc_param_selection(taiwan_x_gridsearch, taiwan_y_gridsearch,5)
# print(best_params)
# stop = timeit.default_timer()
# print('Time: ', stop - start)

Training Dataset size effect


train_size_scores = np.ndarray((0,4))


steps = 10

for i in range (1,steps+1):  
  taiwan_x_cut = taiwan_x_train[:int(i/steps*len(taiwan_x_train))]
  taiwan_y_cut = taiwan_y_train[:int(i/steps*len(taiwan_y_train))]  

  #print(np.shape(taiwan_x_cut))
  #print(np.shape(taiwan_y_cut))

  # Create SVM classifier with best hyperparameters
  SVM = SVC(C = 10, gamma=1, kernel='rbf')
  # Fit training Data to Classifier
  SVM.fit(taiwan_x_cut, taiwan_y_cut)
  # Evaluate Classifier Performance on Validation Set
  scores = score_model(SVM, taiwan_x_val, taiwan_y_val)
  train_size_scores = np.append(train_size_scores, np.append(np.reshape(i*100/steps, (1,1)), scores, axis=1), axis=0)

df=pd.DataFrame(train_size_scores)
df.to_csv(drive_path+'Evaluation/eval_svm_data_size.csv', header=False, sep=';', decimal=',')




plt.figure()
plt.plot(train_size_scores[:,[0]], train_size_scores[:,[1]])
plt.title('SVM Performance / Training Data')
plt.xlabel('Dataset Size / %')
plt.ylabel('Accuracy')
plt.show()

ROC Curve and AUROC

# # ROC CURVE and AUROC

# from sklearn.metrics import roc_curve, roc_auc_score, auc

# def plot_roc_curve(models, models_name, y_test, x_test):
#     plt.figure()
#     colors = ["green","red","cyan","magenta","yellow"]

#     for index,model in enumerate(models):

#       fpr = dict()
#       tpr = dict()

   

#       # For discrete binary classifiers such as SVM, regular predictions will directly output the class (0 or 1). 
#       # To provide a meaningful ROC curve with probabilities outputs and thresholds, the model property "probability" need to be set as "True" before training.
#       if models_name[index] == "SVM":
#         y_pred = model.predict_proba(x_test)[:,1]
#       else :
#         y_pred = model.predict(x_test)

#       for i in range(2):
#           fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
        

#       auroc = roc_auc_score(y_test, y_pred)
#       plt.plot(fpr[1], tpr[1],color=colors[index], label="%s (AUC = %0.2f)"  % (models_name[index],auroc))


#     plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     plt.show()


# plot_roc_curve([SVM,LinearSVM],["SVM",'LinearSVM'],taiwan_y_test, taiwan_x_test)

Gains Curve and Lift Curve

import scikitplot as skplt
import matplotlib.pyplot as plt
y_pred_SVM= SVM.predict(taiwan_x_test)
# plot functions need "onTe hot encoded" class probabilities
y_neg = 1 - y_pred_SVM
y_both = np.append(np.reshape(y_neg, (-1,1)), np.reshape(y_pred_SVM, (-1,1)), axis=1)
# plot curves
skplt.metrics.plot_cumulative_gain(taiwan_y_test, y_both)
plt.show()
skplt.metrics.plot_lift_curve(taiwan_y_test, y_both)
plt.show()

Confusion Matrix

skplt.metrics.plot_confusion_matrix(taiwan_y_test, np.argmax(y_both, axis=1), normalize=True)
plt.show()

### Model 4 - Neural Network

DATA & MODULES IMPORTATION

# Mount Google Drive
from google.colab import drive
drive.mount('/gdrive')

# Define Location of Project Folder
# Define Location of Project Folder
drive_path = '/gdrive/My Drive/NTU/FinTech/Fin Tech: Final presentation/Final Project/'

taiwan_data = 'Datasets/UCI_Credit_Card/UCI_Credit_Card.csv'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
import keras
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import confusion_matrix
#Importing data
taiwan_raw = pd.read_csv(drive_path + taiwan_data)

DATA SPLITTING & NORMALIZATION

# Fetch Label 
taiwan_y = taiwan_raw['default.payment.next.month']
# Remove ID and Label from Dataset
taiwan_raw.drop(columns='default.payment.next.month', inplace=True)
taiwan_raw.drop(columns='ID', inplace=True)


taiwan_x_train, taiwan_x_temp, taiwan_y_train, taiwan_y_temp = train_test_split(taiwan_raw, taiwan_y, test_size=0.2)
taiwan_x_val, taiwan_x_test, taiwan_y_val, taiwan_y_test = train_test_split(taiwan_x_temp, taiwan_y_temp, test_size=0.1/0.2)

# Rescale Data between 0...1
taiwan_min = np.min(taiwan_x_train)
taiwan_max = np.max(taiwan_x_train)
taiwan_x_train = (taiwan_x_train-taiwan_min) / (taiwan_max-taiwan_min)
taiwan_x_val = (taiwan_x_val-taiwan_min) / (taiwan_max-taiwan_min)
taiwan_x_test = (taiwan_x_test-taiwan_min) / (taiwan_max-taiwan_min)

CONVERSION DF --> ARRAYS

#To array
taiwan_x_train = np.array(taiwan_x_train)
taiwan_y_train = np.array(taiwan_y_train)
taiwan_x_val = np.array(taiwan_x_val)
taiwan_x_test = np.array(taiwan_x_test)
taiwan_y_val = np.array(taiwan_y_val)
taiwan_y_test = np.array(taiwan_y_test)

print("Train_shape = ", taiwan_x_train.shape)
print("Val_shape = ", taiwan_x_val.shape)
print("Test_shape = ", taiwan_x_test.shape)

CATEGORIZING TARGETS

def categorize(target):
  Y = []
  A = [0.,1.]
  B = [1.,0.]
  for i in range(len(target)):
    if target[i]==1:
      Y.append(A)
    elif target[i]==0:
      Y.append(B)
  return np.array(Y)

taiwan_y2_train = categorize(taiwan_y_train)
taiwan_y2_val = categorize(taiwan_y_val)
taiwan_y2_test = categorize(taiwan_y_test)

MODEL 0 : NO DOWNSAMPLING

## NN creation
model0 = Sequential()
#Add the layers
#First layer
model0.add(Dense(60, activation = 'relu'))
#Second layer
model0.add(Dense(25, activation= 'relu'))
model0.add(Dense(45, activation= 'relu'))
#final output layer
model0.add(Dense(2, activation= 'softmax'))
#Compile
optimizer = Adam(lr = 0.01)
model0.compile(loss = 'categorical_crossentropy', optimizer ='Adam', metrics = ['accuracy'])

#model training with early stopping
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#history with train and val
history = model0.fit(taiwan_x_train, taiwan_y2_train ,verbose = 2, epochs = 25, batch_size = 1, validation_data = (taiwan_x_val, taiwan_y2_val), callbacks = [callback])

#plot curves
loss_curve = history.history["loss"]

loss_val_curve = history.history["val_loss"]

plt.plot((loss_curve), label="Train")
plt.plot((loss_val_curve), label="Val")
plt.legend(loc='upper left')
plt.title("Loss")
plt.grid()

plt.show()

print("Accuracy on train = ", model0.evaluate(taiwan_x_train, taiwan_y2_train)[1])
print("Accuracy on validation = ", model0.evaluate(taiwan_x_val, taiwan_y2_val)[1])
print("Accuracy on test = ", model0.evaluate(taiwan_x_test, taiwan_y2_test)[1])

DOWNSAMPLING ATTEMPT

We randomly delete members of the non_default class among the training set, which outnumbers the population of default subjects.

def downsample(trainX, trainY):

  print('Proportion of positive labels to equalize : ', np.sum(trainY)/len(trainY))
  eject_prob = 1-np.sum(trainY)/len(trainY)
  
  X = trainX
  Y = trainY
  indexes_to_delete = []

  for i in range(len(trainX)):
    if trainY[i] ==0: #this is the larger class'label : we want to reduce the individuals number in that class
      p = np.random.uniform()
      if p <= eject_prob:
        indexes_to_delete.append(i)

  X = np.delete(X, indexes_to_delete, axis = 0)
  Y = np.delete(Y, indexes_to_delete, axis = 0)
  print(len(indexes_to_delete), ' values deleted')
  print('New label balance : ', np.sum(Y)/len(Y))
  print('Size reduction : Original ', len(trainY), ' -->  ',len(Y))

  return X, Y

taiwan_x3_train, taiwan_y3_train = downsample(taiwan_x_train, taiwan_y_train)
taiwan_y3_train = categorize(taiwan_y3_train)

print(taiwan_x3_train.shape)
print(taiwan_y3_train.shape)

MODEL DS : TRAINED ON DOWNSAMPLED/ BALANCED DATA

## NN WITH DOWNSAMPLING
model_ds = Sequential()
#Add the layers
#First layer
model_ds.add(Dense(60, activation = 'relu'))
#Second layer
model_ds.add(Dense(25, activation= 'relu'))
model_ds.add(Dense(45, activation= 'relu'))
#final output layer
model_ds.add(Dense(2, activation= 'softmax'))
#Compile
optimizer = Adam(lr = 0.01)
model_ds.compile(loss = 'categorical_crossentropy', optimizer ='Adam', metrics = ['accuracy'])

#model training with early stopping
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#history with train and val
history = model_ds.fit(taiwan_x3_train, taiwan_y3_train ,verbose = 2, epochs = 50, batch_size = 1, validation_data = (taiwan_x_val, taiwan_y2_val), callbacks = [callback])

#plot curves
loss_curve = history.history["loss"]
loss_val_curve = history.history["val_loss"]

plt.plot((loss_curve), label="Train")
plt.plot((loss_val_curve), label="Val")
plt.legend(loc='upper left')
plt.title("Loss")
plt.grid()

plt.show()

print("Accuracy on DOWNSAMPLED train = ", model_ds.evaluate(taiwan_x3_train, taiwan_y3_train)[1])
print("Accuracy on NON DOWNSAMPLED train = ", model_ds.evaluate(taiwan_x_train, taiwan_y2_train)[1])
print("Accuracy on validation = ", model_ds.evaluate(taiwan_x_val, taiwan_y2_val)[1])
print("Accuracy on test = ", model_ds.evaluate(taiwan_x_test, taiwan_y2_test)[1])

REPARTITION OF BAD INDIVIDUALS

Here we analyze the distance between the true classes and the prediction for each individual.

For example : if the individual A true label is 1 (default), we'll look at the predicted value of class 1 (the default class, value observed at this point should be high is prediction is accurate).

We then measure the difference (absolute value of distance) between the true label and the predicted value of this label. We then rank the individual in fuction of this distance: if the label is 1 (default) and the predicted value of class 1 (default class) is 0.95, the distance is 0.05. In this case we add the individual to the group having a distance to truth between 0 and 0.1.

In all, we have 10 groups, gathering individuals depending of their distance by intervals of 0.1.

Finally, we compute the proportion of each group regarding the sum of all individuals sorted and belonging to one same class.

#We create a custom display which counts the number of TRUE defaults predicted per decile

def study_prediction(model, X, Y):
  import matplotlib.pyplot as plt
  prediction = model.predict(X)
  class0_pred = prediction[:,0]
  class1_pred = prediction[:,1]

  groups = ['0,0.1', '0.1,0.2','0.2,0.3','0.3,0.4', '0.4,0.5', '0.5,0.6', '0.6,0.7', '0.7,0.8','0.8,0.9', '0.9,1']
  x = np.arange(len(groups))
  group_01=0
  group_02=0
  group_03=0
  group_04=0
  group_05=0
  group_06=0
  group_07=0
  group_08=0
  group_09=0
  group_10=0
  for i in range(len(Y)):
    if Y[i]==1:
      a = abs(1 - class1_pred[i])
      if a <=0.1:
        group_01 += 1
      elif 0.1<a<=0.2:
        group_02 +=1
      elif 0.2<a<=0.3:
        group_03 +=1
      elif 0.3<a<=0.4:
        group_04 +=1
      elif 0.4<a<=0.5:
        group_05 +=1
      elif 0.5<a<=0.6:
        group_06 +=1
      elif 0.6<a<=0.7:
        group_07 +=1
      elif 0.7<a<=0.8:
        group_08 +=1
      elif 0.8<a<=0.9:
        group_09 +=1
      elif 0.9<a:
        group_10 +=1

  group1_repartition = [group_01, group_02, group_03, group_04, group_05, group_06, group_07, group_08, group_09, group_10]
  group1_proportion = group1_repartition/np.sum(group1_repartition)

  print('True default proportion having distance < 0.5 = ', np.sum(group1_proportion[0:5]))

  group_01=0
  group_02=0
  group_03=0
  group_04=0
  group_05=0
  group_06=0
  group_07=0
  group_08=0
  group_09=0
  group_10=0
  for i in range(len(Y)):
    if Y[i]==0:
      a = abs(1-class0_pred[i])
      if a <=0.1:
        group_01 += 1
      elif 0.1<a<=0.2:
        group_02 +=1
      elif 0.2<a<=0.3:
        group_03 +=1
      elif 0.3<a<=0.4:
        group_04 +=1
      elif 0.4<a<=0.5:
        group_05 +=1
      elif 0.5<a<=0.6:
        group_06 +=1
      elif 0.6<a<=0.7:
        group_07 +=1
      elif 0.7<a<=0.8:
        group_08 +=1
      elif 0.8<a<=0.9:
        group_09 +=1
      elif 0.9<a:
        group_10 +=1

  group0_repartition = [group_01, group_02, group_03, group_04, group_05, group_06, group_07, group_08, group_09, group_10]
  group0_proportion = group0_repartition/np.sum(group0_repartition)
  print('True non default porportion having distance < 0.5 = ', np.sum(group0_proportion[0:5]))

  width = 0.35
  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width/2, group0_proportion, width, label='True Non Default')
  rects2 = ax.bar(x + width/2, group1_proportion, width, label='True Default')

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Proportion per Category of individuals')
  ax.set_title('Distance between prediction and truth')
  ax.set_xticks(x)
  ax.set_xticklabels(groups)
  ax.legend()
  plt.grid()
  return plt.show()


study_prediction(model0, taiwan_x_test, taiwan_y_test)
study_prediction(model_ds, taiwan_x_test, taiwan_y_test)

LIFT CURVES, ACCURACY, SENSITIV., SPECIF.

*Gain Curves*

pip install scikit.plot

import matplotlib.pyplot as plt
import scikitplot as skplt


skplt.metrics.plot_cumulative_gain(taiwan_y_test, model0.predict(taiwan_x_test))
plt.title('Gain Curve On Test Set With No Downsampling Model')
plt.show()

skplt.metrics.plot_cumulative_gain(taiwan_y_test, model_ds.predict(taiwan_x_test))
plt.title('Gain Curve On Test Set With Downsampling Model')
plt.show()

*Scores*

import numpy as np
from sklearn.metrics import confusion_matrix

def score_model(model, x, y):
    #scores = model.evaluate(x, y, verbose=0)  
    y_pred = model.predict(x)[:,1]
    y_pred = np.where(y_pred > 0.5, 1, 0)
    conf_matrix = confusion_matrix(y_true=y, y_pred=y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    sensitivity = tp / (tp+fn)
    specificity = tn / (fp+tn)

    

    #print("Accuracy: %.2f Sensitivity: %.2f Specificity: %.2f" % (accuracy, sensitivity, specificity))
    return np.reshape( (accuracy, sensitivity,specificity) , (1,3) )  

# Final Prediction of Test Dataset
test_scores = np.ndarray((0,4))
# Evaluate Classifier Performance on Test Set, MODEL 0
scores = score_model(model0, taiwan_x_test, taiwan_y_test)
print('Basic (NO DOWNSAMPLING) Network socres : ', scores)

# Evaluate Classifier Performance on Test Set, MODEL 1 (DOWNSAMPLING)
scores = score_model(model_ds, taiwan_x_test, taiwan_y_test)
print('Basic (NO DOWNSAMPLING) Network socres : ', scores)

*Confusion Matrix*

y_pred_0 = model0.predict(taiwan_x_test)[:,1]
y_pred_0 = np.where(y_pred_0 > 0.5, 1, 0)

y_pred_ds = model_ds.predict(taiwan_x_test)[:,1]
y_pred_ds = np.where(y_pred_ds > 0.5, 1, 0)

skplt.metrics.plot_confusion_matrix(taiwan_y_test,y_pred_0, normalize=True)
plt.title('Confusion matrix Model 0')
plt.show()

skplt.metrics.plot_confusion_matrix(taiwan_y_test, y_pred_ds, normalize=True)
plt.title('Confusion matrix Model 1 (Downsampling)')
plt.show()

### Evaluation / Comparison

ROC Curve / AUC comparison

# ROC CURVE and AUROC

from sklearn.metrics import roc_curve, roc_auc_score, auc

### Decision tree
DT = tree.DecisionTreeClassifier(min_samples_leaf= 10, max_depth= 4)
DT = clf.fit(taiwan_x_train, taiwan_y_train)


def plot_roc_curve(models, models_name, y_test, x_test):
    plt.figure()
    colors = ["green","red","cyan","magenta","yellow"]

    for index,model in enumerate(models):

      fpr = dict()
      tpr = dict()

   

      # For discrete binary classifiers such as SVM, regular predictions will directly output the class (0 or 1). 
      # To provide a meaningful ROC curve with probabilities outputs and thresholds, the model property "probability" need to be set as "True" before training.
      if models_name[index] == "SVM":
        y_pred = model.predict_proba(x_test)[:,1]
      else :
        y_pred = model.predict(x_test)

      for i in range(2):
          fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
        

      auroc = roc_auc_score(y_test, y_pred)
      plt.plot(fpr[1], tpr[1],color=colors[index], label="%s (AUC = %0.2f)"  % (models_name[index],auroc))


    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


plot_roc_curve([KNN,DT,SVM,LinearSVM],["KNN","DT","SVM",'NN',],taiwan_y_test, taiwan_x_test)