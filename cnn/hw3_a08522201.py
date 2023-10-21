#import the libraries
from keras.datasets import fashion_mnist
import tensorflow as tf

import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



##### DATA PREPROCESSING

# Import : Fashion MNIST
print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
X = np.concatenate((trainX, testX))
Y = np.concatenate((trainY,testY))

# Reshape the dataset and convert to float
X = X.reshape(-1, 784)
X = X.astype(float)
Y = Y.astype(np.int32)

# Split the data to the ratio of 0.8
trainX = X[0:56000,:]
testX = X[56000:-1,:]
trainY = Y[0:56000]
testY = Y[56000:-1]

# Create Hot-Encoder for the labels
trainY = keras.utils.to_categorical(trainY, 10, np.int32)
testY = keras.utils.to_categorical(testY, 10, np.int32)

#Normalisation of X
trainX = trainX.reshape(trainX.shape[0],28,28,1)
testX = testX.reshape(testX.shape[0],28,28,1)
trainX = trainX/(trainX.shape[1]*trainX.shape[2])
testX = testX/(testX.shape[1]*testX.shape[2])



##### QUESTION 1

def create_model(neuronsHL=10, filter_size = 1, stride_size = 1, kernel_size = 3):
    """creation of the CNN model with 2 convolutions layers"""
    model = Sequential()

    #1 = convolution + max pooling + dropout
    model.add(Conv2D(filter_size,strides = stride_size, kernel_size= kernel_size, activation= 'relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Dropout(0.3))

    #2 = convolution + max pooling + dropout
    model.add(Conv2D(filter_size, strides = stride_size, kernel_size=kernel_size, activation= 'relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Dropout(0.3))

    #3 = Flatten + simple layer
    model.add(Flatten())
    model.add(Dense(neuronsHL, activation= 'relu'))

    #4 = final output layer
    model.add(Dense(10, activation= 'softmax'))

    #Compilation of the model
    model.compile(  loss = 'categorical_crossentropy',
                    optimizer ='Adam',
                    metrics = ['accuracy'])
    model.summary()

    return(model)

def Optiparameters():
    """Return the optimal parameters among all the parameter grid"""
    # create and initialise the model with values
    model = KerasClassifier(build_fn=create_model, verbose=2)

    # define the grid search parameters
    neuronsHL = [64]
    filter_size = [12, 36, 48]
    stride_size = [1, 2]
    kernel_size = [3, 4]

    #Grid_Search
    param_grid = dict(
        neuronsHL = neuronsHL,
        epochs=[25],
        filter_size = filter_size,
        stride_size = stride_size,
        kernel_size = kernel_size)

    grid = GridSearchCV(estimator = model, param_grid= param_grid ,n_jobs = -1, cv=3, verbose = 2)
    grid_result = grid.fit(trainX[:5000], trainY[:5000]) # grid search on a small data part to reduce its cost
    #print the results
    print("Thanks to the optimisation we got this optimal results :")
    print("Best score in search : ", grid_result.best_score_,
        "\nParameters used :",  grid_result.best_params_,"\n")
    return( grid_result.best_params_['neuronsHL'],
            grid_result.best_params_['filter_size'],
            grid_result.best_params_['stride_size'],
            grid_result.best_params_['kernel_size'])

# Research of the optimal parameters
print("\n### QUESTION 1 :")
print("research of the optimal parameters\n")
# (filter_number, filter_size, std) = Optiparameters()



##### QUESTION 2

def Training(model):
    """Train the input model with 25 epochs"""
    print("[INFO] Training of the CNN with this parameters:")
    print("  neuronsHL = "+str(neuronsHL)+"\n  filter_size = "+str(filter_size)+"\n  stride_size = "+str(stride_size)+"\n  kernel_size = "+str(kernel_size))
    history = model.fit(trainX, trainY,
                        batch_size= 50,
                        epochs=25,
                        validation_data=(testX, testY),
                        verbose = 2)
    return(model, history)

def plot_curves(history):
    """Trace learning curve/accuracy rate"""
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


print("\n### QUESTION 2 :")
# Training of the model with the parameters found before :
neuronsHL = 63
filter_size = 48
stride_size = 1
kernel_size = 3
model = create_model(neuronsHL, filter_size, stride_size , kernel_size)
(model, history) = Training(model)

# Plot the learning curve and accuracy rate curve:
print("Plot the learning curve and accuracy rate curve of the model\n")
plot_curves(history)



##### QUESTION 3
labelNames = ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def plot_activation(model):
    """plot the activation images for a random sample"""

    # Creates model with the fisrt layer only
    activation_model = keras.Model(inputs=model.input, outputs=model.layers[0].output)
    # activation_model.summary()

    # Choose a random sample
    num = int(np.random.randint(low=1, high=testX.shape[0], size=1))
    sample = trainX[num].reshape(1,28,28,1)

    #Get the true label and prediction of the sample
    result_1stlayer = activation_model.predict(sample)
    result_1stlayer = result_1stlayer[0]
    label = model.predict(sample)
    label = np.argmax(label)
    true_label = np.argmax(trainY[num])

    #Control if the prediction is correct
    pred = "FALSE"
    if label == true_label:
        pred = "TRUE"

    # Plot the sample image
    plt.imshow(np.reshape(trainX[num],(28,28)), cmap=plt.cm.binary)
    plt.title("INPUT SAMPLE: Prediction = "+ labelNames[label]+" {"+pred+"}", fontsize=12,weight = 'bold')
    plt.xticks([])
    plt.yticks([])

    # Plot the result of 1st layer
    plt.figure(figsize=(6,8))
    for i in range(48):
        plt.subplot(6,8,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(result_1stlayer[:,:,i], cmap=plt.cm.binary)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

# Plot a random sample and its activation images
print("\n### QUESTION 3 :")
print("Plot a random sample with its 48 activation images\n")
plot_activation(model)



##### QUESTION 4

def plot_result(model):
    """Plot the result of 16 random samples"""

    #generation of the random samples
    num = np.random.choice(np.arange(0, len(testY)), size=(16,))
    sample = np.zeros((16,28,28,1))
    true_label = np.zeros(16)
    for i in range(16):
        sample[i] = testX[num[i]]
        true_label[i] = int(np.argmax(testY[num[i]]))

    #Apply the model
    label = model.predict(sample)
    predicted_label = np.zeros(16)
    for i in range(label.shape[0]):
        predicted_label[i] = np.argmax(label[i])

    #reshape images
    sample = sample.reshape(16,28,28)

    #plot the image
    fig, axs = plt.subplots(4,4,figsize=(6,6))
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.suptitle('Result of the CNN (16 samples)', fontsize=16, va='top',weight = 'bold')
    k=0
    for i in range(4):
        for j in range(4):
            axs[i,j].yaxis.set_major_locator(plt.NullLocator())
            axs[i,j].xaxis.set_major_locator(plt.NullLocator())
            if (predicted_label[k] == true_label[k]):
                font = {'family': 'sans-serif',
                        'color':  'lime',
                        'weight': 'bold',
                        'size': 12,
                        }
                axs[i,j].text(0,5,labelNames[int(predicted_label[k])], fontdict=font)
            else:
                font = {'family': 'sans-serif',
                        'color':  'red',
                        'weight': 'bold',
                        'size': 12,
                        }
                axs[i,j].text(0,5,labelNames[int(predicted_label[k])], fontdict=font)
            axs[i,j].imshow(sample[k], cmap=plt.get_cmap('gray'))
            k+=1
    plt.show()


# Plot of the result for 16 samples
print("\n### QUESTION 4:")
print("Plot of the result for 16 samples \nPrediction: Green = correct / red = incorrect\n")
plot_result(model)

### CONCLUSION ON THE PRECISION


# make predictions on the test set
preds = model.predict(testX)

# show a nicely formatted classification report
print("\nCONCLUSION ON PRECISION:")
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
	target_names=labelNames))


