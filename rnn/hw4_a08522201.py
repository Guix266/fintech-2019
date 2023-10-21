
"""
fintech hw4 : RNN
Created on Sun Dec 22 19:48:12 2019
@author: Guillaume DESERMEAUX a08522201
"""
# Import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

#import the 2 libraries
import mpl_finance
import talib

from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Dropout, GRU
from sklearn.model_selection import train_test_split
import matplotlib.dates as dates

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mpl.style.use('default')


# Load the dataset and split it into training / testing sets
file = 'SPY.csv'
df = pd.read_csv("SPY.csv")
    
# =============================================================================
# I) Plot Charts from 2018/1/1 to 2018/12/31
# =============================================================================

def plot_charts():
    # data processing
    ohlc_df = df.loc[6044:,:].copy()
    ohlc_df = ohlc_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    ohlc_df['Date'] = ohlc_df['Date'].map(dates.datestr2num)
    # create the figure
    fig, ax = plt.subplots(3, figsize=(20, 20), gridspec_kw={'height_ratios': [6, 2, 3]})
    # plt.subplots_adjust(wspace=0, hspace=0)
    
    for i in range(0,3):
        ax[i].xaxis_date()
        ax[i].set_xlim(left = 736685, right = 737065)
    
    ax[2].xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d"))
    
    plt.xlabel("Date", fontsize=12)
    
    ### a) Candlestick chart
    
    # plt.text(28, 230, 'Candlestick chart with 2 moving average line', fontsize=16, weight = 'bold')
    
    
    # Trace the moving averages
    MA1 = talib.MA(df.loc[6014:,'Close'].values, timeperiod=10, matype=0)
    MA2 = talib.MA(df.loc[6014:,'Close'].values, timeperiod=30, matype=0)
    
    ax[0].plot(ohlc_df['Date'].values , MA1[30:], linewidth=0.5, label = "MA = 10 days")
    ax[0].plot(ohlc_df['Date'].values , MA2[30:], linewidth=0.5, label = "MA = 30 days")
    ax[0].legend(fancybox = True)
    
    # Trace the candles
    mpl_finance.candlestick_ohlc(ax[0], ohlc_df.values, 
                                  width=0.3, colorup='g', colordown='r', alpha=1)
    
    
    ### b) KD line
    
    slowk, slowd = talib.STOCH(ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"])
    
    ax[1].plot(ohlc_df['Date'].values, slowk, label = "K")
    ax[1].plot(ohlc_df['Date'].values, slowd, label = "D")
    ax[1].legend(fancybox = True)
    
    
    ### c) Volume chart
    
    ax[2].set_ylim(bottom = 0,top = 8000000000)
    ax[2].apply_aspect()
    mpl_finance.volume_overlay3(ax[2],ohlc_df.values, colorup='g', colordown='r', width=3, alpha=1.0)
    
    # plt.savefig('finance_chart.png', dpi=400)

plot_charts()

# =============================================================================
# II) Data processing
# =============================================================================


# Add moving averages
df["Moving_av_10"] = talib.MA(df['Close'].values, timeperiod=10, matype=0)
df["Moving_av_30"] = talib.MA(df['Close'].values, timeperiod=30, matype=0)
# Add K and D
df["K"], df["D"] = talib.STOCH(df["High"], df["Low"], df["Close"])

# Add Relative Strength Index
df["RSI_9"] = talib.RSI(df['Close'].values, timeperiod=9)
df["RSI_14"] = talib.RSI(df['Close'].values, timeperiod=14)

# Add Exponential moving average
df["Exp_moving_av"] = talib.EMA(df['Close'].values)

# Add Moving Average Convergence/Divergence
(df["macd"], a, b) = talib.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)

# Normalization of the data
columns = list(df.columns.values)
columns.remove('Date')
# columns.remove('Close')

for col in columns:
    mini = (df.loc[:5793,col]).min()
    maxi = (df.loc[:5793,col]).max()
    df[col] = (df[col] - mini)/(maxi - mini)

# Replace the Nan with 0
df = df.where((pd.notna(df)),0)

# print(df)
# print("max = \n" + str(df.max()))
# print("min = \n" + str(df.min()))

# =============================================================================
# III) Creation of the RNN input
# =============================================================================

def create_input(df, time_step = 30):
    """create an imput of 3 dimentions according to the required imput of the model"""
    
    # create the vector date of validation training
    date_valid = df.loc[5793:,'Date'].map(dates.datestr2num)

    # Separation od the data into training and testing dataset
    Y = df.loc[:,["Close"]]
    X = df.drop(["Date"], axis=1)
    
    # transfor to array
    X = np.array(X)
    Y = np.array(Y)
    
    input_dim = X.shape[1]
    # Size of each chunk
    num_chunk = X.shape[0] - (time_step - 1)
    # create the adapted array
    new_X = np.zeros((num_chunk, time_step, input_dim))
    new_Y = Y[(time_step - 1):]
    for i in range(num_chunk):
        new_X[i] = X[i:i+time_step,:]
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(new_X, new_Y, train_size = 5764, shuffle=False)   
    
    return(X_train, X_valid, Y_train, Y_valid, np.array(date_valid))


(X_train, X_valid, Y_train, Y_valid, date_valid) = create_input(df)
# print(X_train)
# print(Y_train)


# =============================================================================
# IV) RNN model with SimpleRNN
# =============================================================================

def create_simpleRNN_model(batch = 15):
    model = keras.Sequential()
    # Add the RNN layer
    
    model.add(SimpleRNN( units =  100 ))
    model.add(Dropout(0.3))

    model.add(Dense( units = 1))
    
    model.compile(  loss = "mean_squared_error", optimizer ='Adam')
    
    history = model.fit(X_train, Y_train, batch_size= batch, epochs=15,
                        validation_data = (X_valid, Y_valid))

    return(model, history)


model1, history1 = create_simpleRNN_model(batch = 20)
Y_pred1 = model1.predict(X_valid)

# =============================================================================
# V) Plot loss curve / actual and predict 'Close' value
# =============================================================================

def plot_loss(history):
    # Get the values
    loss_curve = history.history["loss"]
    loss_val_curve = history.history["val_loss"]

    #Print them on 2 graphs
    plt.plot(loss_curve, label="Training")
    plt.plot(loss_val_curve, label="Validation")
    plt.legend(frameon=False, loc='upper center', ncol=2)
    plt.xlabel('epochs')
    plt.ylabel('MODEL LOSS')
    plt.show()

def plot_prediction(date, Y_pred, Y_true):
    """Plot both predicted and real curve"""
    
    # create the figure
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_title('Closing Value of the stock : Real and Predicted', fontsize=18, weight = 'bold')
    plt.xlabel("Date", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    
    ax.plot(date, Y_true, "b-" , linewidth=0.5, label = "real")
    ax.plot(date, Y_pred, "r--", linewidth=0.5, label = "predicted")
    ax.legend(fontsize = 13)
    
    #ax.set_xlim(left = 736685, right = 737065)       
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    

# Plot loss function
plot_loss(history1)

# Plot presiction and real curve
plot_prediction(date_valid, Y_pred1, Y_valid)

# =============================================================================
# VI) RNN model with LSTM cell
# =============================================================================

def create_LSTM_model(batch = 30):
    model = keras.Sequential()
    # Add the RNN layer
    
    model.add(LSTM( units =  75 ))
    model.add(Dropout(0.3))
    
    model.add(Dense( units = 1))
    
    model.compile(  loss = "mean_squared_error", optimizer ='Adam')
    
    history = model.fit(X_train, Y_train, batch_size= batch, epochs=15,
                        validation_data = (X_valid, Y_valid))

    return(model, history)


model2, history2 = create_LSTM_model(batch = 20)
Y_pred2 = model2.predict(X_valid)

# Plot loss function
plot_loss(history2)

# Plot presiction and real curve
plot_prediction(date_valid, Y_pred2, Y_valid)

# =============================================================================
# VII) RNN model with GRU cell
# =============================================================================

def create_GRU_model(batch = 30):
    model = keras.Sequential()
    # Add the RNN layer
    
    model.add(GRU( units =  100 ))
    model.add(Dropout(0.3))
    
    model.add(Dense( units = 1))
    
    model.compile(  loss = "mean_squared_error", optimizer ='Adam')
    
    history = model.fit(X_train, Y_train, batch_size= batch, epochs=15,
                        validation_data = (X_valid, Y_valid))

    return(model, history)


model3, history3 = create_LSTM_model(batch = 20)
Y_pred3 = model3.predict(X_valid)

# Plot loss function
plot_loss(history3)

# Plot presiction and real curve
plot_prediction(date_valid, Y_pred3, Y_valid)

# =============================================================================
# VIII) Comparison of the models
# =============================================================================

def plot_comparison(date, Y_pred1, Y_pred2, Y_pred3, Y_true):
    """Plot both predicted and real curve"""
    
    # create the figure
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_title('Closing Value of the stock : Real and Predicted', fontsize=18, weight = 'bold')
    plt.xlabel("Date", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    
    ax.plot(date, Y_pred1, "g-." , linewidth=0.5, label = "Predicted: simpleRNN")
    ax.plot(date, Y_pred2, "c-." , linewidth=0.5, label = "Predicted: LSTM")
    ax.plot(date, Y_pred3, "m-." , linewidth=0.5, label = "Predicted: GRU")
    ax.plot(date, Y_true, "b-", linewidth=1, label = "Real")
    ax.legend(fontsize = 13)
      
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.savefig('prediction_comparison.png', dpi=400)

plot_comparison(date_valid, Y_pred1, Y_pred2, Y_pred3, Y_valid)