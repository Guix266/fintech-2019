#### HOMEWORK 2 Guillaume DESERMEAUX A08522201
# Matplotlib
import matplotlib.pyplot as plt
# Tensorflow + keras
import tensorflow as tf
from tensorflow import keras as kr
# Numpy and Pandas
import numpy as np
import pandas as p
# Ohter import
import sys
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

##### QUESTION 1
##i)

# Load the dataset
file = 'Data.csv'
df = p.read_csv(file)
#print(df.shape)
df =df.sample(frac=1)

#Data Normalisation
df2=df.copy()
trainset=df.head(1120)
m=trainset.mean()
e=trainset.std()
df=(df-m)/e

#Creation of training and validation dataset
data = np.array(df)
data2 = np.array(df2)
Xtrain = data[0:1120,0:30]
Ytrain = data2[0:1120,30]
Xvalid = data[1120:,0:30]
Yvalid = data2[1120:,30]

# Creation of the model :

def Creation_of_network(nb_hidlayer, nb_hidunits, learning_rate):
    """Create the neural network corresponding to the parameters"""
    model = kr.models.Sequential()
    # Add the layers
    for i in range(0,nb_hidlayer):
        model.add(kr.layers.Dense(nb_hidunits, input_shape=(30,) ,activation="relu"))
    model.add(kr.layers.Dense(2, activation="softmax"))

    # Compile the model
    Adam = kr.optimizers.Adam(lr=learning_rate)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam,
        metrics=["accuracy"])
    return(model)

def Creation_of_network2(nLayer_1, nLayer_2, learning_rate):
    """Create the neural network corresponding to the parameters"""
    model = kr.models.Sequential()
    # Add the layers
    model.add(kr.layers.Dense(nLayer_1, input_shape=(30,) ,activation="relu"))
    model.add(kr.layers.Dense(nLayer_2, input_shape=(30,) ,activation="relu"))
    model.add(kr.layers.Dense(2, activation="softmax"))

    # Compile the model
    Adam = kr.optimizers.Adam(lr=learning_rate)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam,
        metrics=["accuracy"])
    return(model)


def Training_model(nb_hidlayer, nb_hidunits, learning_rate, epochs, batch_size):
    """Train the model with the following batch_size"""
    model=Creation_of_network(nb_hidlayer, nb_hidunits, learning_rate)
    history = model.fit(Xtrain, Ytrain,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(Xvalid,Yvalid))
    return(model, history)

def Training_model2(batch_size,epochs,learning_rate,nLayer_2,nLayer_1):
    model=Creation_of_network2(nLayer_1, nLayer_2, learning_rate)
    history = model.fit(Xtrain, Ytrain,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(Xvalid,Yvalid))
    return(model, history)

def Affiche(history):
    # Get the values
    loss_curve = history.history["loss"]
    acc_curve = history.history["accuracy"]
    loss_val_curve = history.history["val_loss"]
    acc_val_curve = history.history["val_accuracy"]

    #Print them on 2 graphs
    plt.subplot(2,1,1)
    plt.plot(loss_curve, label="Training")
    plt.plot(loss_val_curve, label="Validation")
    plt.legend(frameon=False, loc='upper center', ncol=2)
    plt.xlabel('epochs')
    plt.ylabel('MODEL LOSS')
    plt.subplot(2,1,2)
    plt.plot(acc_curve, label="Training")
    plt.plot(acc_val_curve, label="Validation")
    plt.legend(frameon=False, loc='lower center', ncol=2)
    plt.xlabel('epochs')
    plt.ylabel('MODEL ACCURACY')
    plt.show()

def Optiparameters():
    """Return the optimal parameters among all the parameter grid"""
    # define the grid search parameters
    learning_rate = [0.0001, 0.0005, 0.001]
    batch_size = [3, 5, 10]
    epochs = [50]
    nb_hidunits = [10, 15, 20]
    nb_hidlayer = [1]
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # create and initialise the model with values
    model = KerasClassifier(build_fn=Creation_of_network,
            nb_hidlayer = 1,
            nb_hidunits=1,
            learning_rate=0.01,
            verbose=0)
    #Grid_Search
    param_grid = dict(  batch_size=batch_size,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        nb_hidunits=nb_hidunits,
                        nb_hidlayer=nb_hidlayer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
    grid_result = grid.fit(data[:,0:30], data2[:,30])

    #print of the results
    print("Thanks to the optimisation we got this optimal results :")
    print("Best score in search : ", grid_result.best_score_,
        "\nParameters used :",  grid_result.best_params_,"\n")
    return( grid_result.best_params_['batch_size'],
            grid_result.best_params_['epochs'],
            grid_result.best_params_['learning_rate'],
            grid_result.best_params_['nb_hidlayer'],
            grid_result.best_params_['nb_hidunits'])

def Optiparameters2():
    """Return the optimal parameters among all the parameter grid"""
    # define the grid search parameters
    learning_rate = [0.0001]
    batch_size = [3, 5, 10]
    epochs = [50]
    nLayer_1 = [10, 15, 20]
    nLayer_2 = [10, 15, 20]
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # create and initialise the model with values
    model = KerasClassifier(build_fn=Creation_of_network2,
            nLayer_2 = 1,
            nLayer_1=1,
            learning_rate=0.01,
            verbose=0)
    #Grid_Search
    param_grid = dict(  batch_size=batch_size,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        nLayer_1=nLayer_1,
                        nLayer_2=nLayer_2)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
    grid_result = grid.fit(data[:,0:30], data2[:,30])

    #print of the results
    print("Thanks to the optimisation we got this optimal results :")
    print("Best score in search : ", grid_result.best_score_,
        "\nParameters used :",  grid_result.best_params_,"\n")
    return( grid_result.best_params_['batch_size'],
            grid_result.best_params_['epochs'],
            grid_result.best_params_['learning_rate'],
            grid_result.best_params_['nLayer_2'],
            grid_result.best_params_['nLayer_1'])

# FIRST OPTIMISATION APPROACH
(batch_size,epochs,learning_rate,nb_hidlayer,nb_hidunits)=Optiparameters()

#Application of parameters
# epochs = 50
# batch_size = 3
# learning_rate = 0.0005
# nb_hidlayer = 1
# nb_hidunits = 20
#
# (model, history)=Training_model(nb_hidlayer, nb_hidunits, learning_rate, epochs, batch_size)
# Affiche(history)

# SECOND OPTIMISATION APPROACH
# (batch_size,epochs,learning_rate,nLayer_2,nLayer_1)=Optiparameters2()

#Application of parameters
epochs = 30
batch_size = 3
learning_rate = 0.0001
nLayer_1 = 15
nLayer_2 = 15

(model, history) = Training_model2(batch_size, epochs, learning_rate, nLayer_2, nLayer_1)
Affiche(history)

## ii)

def binary(X):
    y_pred=np.zeros((len(X),1))
    for i in range(0,len(X)):
        if X[i,0]>X[i,1]:
            y_pred[i]=0
        else:
            y_pred[i]=1
    return(y_pred)

def plot_confusion_matrix(y_true, y_pred, title):
    """This function prints and plots the confusion matrix"""
    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    cm = np.array([cm[1],cm[0]])
    #create figure
    plt.imshow(cm, origin = [0,0], aspect ='equal', cmap=plt.cm.Blues)
    plt.title(title)
    plt.xticks([0,1], ['Predicted = 0','Predicted = 1'])
    plt.yticks([0,1], ['True = 1','True = 0'])

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.show()

# Plot for the training
model_output = model.predict(Xtrain)
Ypredt = binary(model_output)
plot_confusion_matrix(Ytrain, binary(model_output),title='Training confusion matrix')
# Plot for the validation
model_output2 = model.predict(Xvalid)
Ypredv = binary(model_output2)
plot_confusion_matrix(Yvalid, binary(model_output2),title='validation confusion matrix')

##iii)

#For the training set
M=[sklearn.metrics.precision_score(Ytrain,Ypredt), sklearn.metrics.recall_score(Ytrain,Ypredt), sklearn.metrics.f1_score(Ytrain,Ypredt),0]
M[3]=(sum(M)/3)
M=np.around(M,decimals = 3)
print("\n   ### NEURAL NETWORK ###")
print("Training set Precision = ", M[0])
print("Training set Recall = ", M[1])
print("Training set f1 score = ", M[2])
print("Average of the 3 metrics = ", M[3],"\n")

# For the validation set
N=[sklearn.metrics.precision_score(Yvalid,Ypredv), sklearn.metrics.recall_score(Yvalid,Ypredv), sklearn.metrics.f1_score(Yvalid,Ypredv), 0]
N[3] = (sum(N)/3)
N=np.around(N,decimals = 3)
print("Validation set Precision = ", N[0])
print("Validation set Recall = ", N[1])
print("Validation set f1 score = ", N[2])
print("Average of the 3 metrics = ", N[3], "\n")

##v)

# Calculation of the decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xtrain, Ytrain)
Y1 = clf.predict(Xvalid)

O = [sklearn.metrics.accuracy_score(Yvalid,Y1), sklearn.metrics.precision_score(Yvalid,Y1), sklearn.metrics.recall_score(Yvalid,Y1), sklearn.metrics.f1_score(Yvalid,Y1)]
O[3] = (sum(O)/3)
O=np.around(O,decimals = 3)

print("   ### DESCISION TREE ###")
print("DecisionTree Acurracy = ",O[0])
print("DecisionTree Precision = ", O[1])
print("DecisionTree  Recall = ", O[2])
print("DecisionTree  f1 score = ", O[3],"\n")

# Calculation of Random forrest

rf = RandomForestClassifier(n_estimators=900)
rf = rf.fit(Xtrain, Ytrain)
Y2 = rf.predict(Xvalid)

P = [sklearn.metrics.accuracy_score(Yvalid,Y2), sklearn.metrics.precision_score(Yvalid,Y2), sklearn.metrics.recall_score(Yvalid,Y2), sklearn.metrics.f1_score(Yvalid,Y2)]
P[3] = (sum(P)/3)
P=np.around(P,decimals = 3)

print("   ### RAMDOM FOREST ###")
print("RandomForest Acurracy = ",P[0])
print("RandomForest Precision = ", P[1])
print("RandomForest  Recall = ", P[2])
print("RandomForest  f1 score = ", P[3],"\n")

##vi)

def learning():
    """Trace the learning curve for NN"""
    train_sizes, train_scores, test_scores = sklearn.model_selection.learning_curve( KerasClassifier(build_fn = Creation_of_network2, nLayer_1=15, nLayer_2=15, learning_rate=0.0001, verbose=0), data[:,0:30], data2[:,30], cv=10, n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def ROC_curve(y_true, y_score):
    fpr, tpr , threshold = sklearn.metrics.roc_curve(y_true, y_score)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def PRC(y_test, y_score):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_score)
    average_precision = sklearn.metrics.average_precision_score(y_test, y_score)
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('Neural Network Precision-Recall curve: AP={0:0.3f}'.format(average_precision))
    plt.show()

data_output = model.predict(data[:,0:30])

##Learning curve
learning()

##ROC_curve
Ypredt = binary(data_output)
ROC_curve(data2[:,30],Ypredt)

##precision-recall curve
Yprb=data_output[:,1]
PRC(data2[:,30],Yprb)

##### QUESTION 2

# Load the dataset
file = 'test_no_Class.csv'
df = p.read_csv(file)
#print(df.shape)
df =df.sample(frac=1)

#Data Normalisation
m=df.mean()
e=df.std()
df=(df-m)/e
data = np.array(df)
#Resolution with the neural network
predicted_class= binary(model.predict(data))

#Write in the txt
file2=open('a08522201_answer.txt','w')
for i in range(len(predicted_class)):
    file2.writelines( "{0:0.0f}".format(predicted_class[i,0]) +'\n')
file2.close()













