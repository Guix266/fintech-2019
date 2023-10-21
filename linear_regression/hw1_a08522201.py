#### HOMEWORK 1 Guillaume DESERMEAUX A08522201
import math as m
import pandas as p
import numpy as np
import matplotlib.pyplot as plt

##### QUESTION 1
##A)
file = 'train.csv'
df = p.read_csv(file)

#Selection of data
Y=df.loc[:,'G3']
Y=Y.to_numpy()
G3train=Y[:800]
G3test=Y[800:1000]

df = df.loc[:,['school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities', 'higher', 'internet', 'romantic', 'famrel',
'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']]

#Replace text by 0 or 1
df=df.replace('GP',1)
df=df.replace('MS',0)
df=df.replace('M',1)
df=df.replace('F',0)
df=df.replace('GT3',1)
df=df.replace('LE3',0)
df=df.replace('yes',1)
df=df.replace('no',0)

#Data Normalisation
trainset=df.head(800)
m=trainset.mean()
e=trainset.std()
for i in range(0,len(df.columns)):
    df.iloc[:,i]=(df.iloc[:,i]-m[i])/e[i]

######TO MIX THE DATA SET
#df =df.sample(frac=1)

trainset=df.head(800)
testset=df.tail(200)

#print(trainset.describe())


##B)

trainset = np.array(trainset.values)
testset = np.array(testset.values)

#To get pseudo-inverse of matrix trainset
pinvTS=np.linalg.pinv(trainset)

#Calculation of optimal weights w
w=np.dot(pinvTS,G3train)

#Calculation of G3test predictions
G3pred = np.dot(testset,w)

#Calculation of RMSE
sum=0
for i in range(0,len(G3test)):
    sum += (G3pred[i]-G3test[i])**2
RMSE=(sum/len(G3test))**(1/2)
print("Model 1, RMSE = " + str(RMSE))


##C)

#Calcultation of optimal weight
X=np.dot(trainset.T,trainset) + len(trainset)*np.identity(trainset.shape[1])/2
A=np.linalg.inv(X)
w2=np.dot(A,np.dot(trainset.T,G3train))

#Calculation of G3 prediction
G3pred2 = np.dot(testset,w2)

#Calculation of RMSE
sum=0
for i in range(0,len(G3test)):
    sum += (G3pred2[i]-G3test[i])**2
RMSE2=(sum/len(G3test))**(1/2)
print("Model 2, RMSE = " + str(RMSE2))


##D

#We had a column to the trainset to get a new weight which will be the bias
trainset2 = np.hstack((trainset,np.ones((trainset.shape[0],1))))
testset2 = np.hstack((testset,np.ones((testset.shape[0],1))))

#Calcultation of optimal weight with Newton Method adding a bias
X=np.dot(trainset2.T,trainset2) + len(trainset)*np.identity(trainset2.shape[1])/2
A=np.linalg.inv(X)
w3=np.dot(A,np.dot(trainset2.T,G3train))

#Calculation of G3 prediction
G3pred3 = np.dot(testset2,w3)

#Calculation of RMSE
sum=0
for i in range(0,len(G3test)):
    sum += (G3pred3[i]-G3test[i])**2
RMSE3=(sum/len(G3test))**(1/2)
print("Model 3, RMSE = " + str(RMSE3))


##E (Baysian regression)

#We get the optimal parameters for the
X=np.dot(trainset2.T,trainset2) + np.identity(trainset2.shape[1])
A=np.linalg.inv(X)
w4=np.dot(A,np.dot(trainset2.T,G3train))

#Calculation of G3 prediction
G3pred4 = np.dot(testset2,w4)

#Calculation of RMSE
sum=0
for i in range(0,len(G3test)):
    sum += (G3pred4[i]-G3test[i])**2
RMSE4=(sum/len(G3test))**(1/2)
print("Model 4, RMSE = " + str(RMSE4))

##F Plot of the functions G3


#plot of predicted notes
plt.plot(G3pred, label = 'Linear regression RMSE='+ str(round(RMSE,2)))
plt.plot(G3pred2, label = 'Linear regression (reg  RMSE='+ str(round(RMSE2,2)))
plt.plot(G3pred3, label = 'Linear regression (reg+biais) RMSE='+ str(round(RMSE3,2)))
plt.plot(G3pred4, label = 'Bayesian linear regression RMSE='+ str(round(RMSE4,2)))
plt.plot(G3test, label= 'Ground Truth')

plt.title("PREDICTED AND REAL G3 GRADES",pad=15)
plt.xlabel("Sample Index")
plt.ylabel("Values (grades)")
plt.legend()
plt.show()


##### QUESTION 2

#Importation of the test_no_G3.csv model
file2 = 'test_no_G3.csv'
df2 = p.read_csv(file2)

#Selection of data

df2 = df2.loc[:,['school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities', 'higher', 'internet', 'romantic', 'famrel',
'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']]

#Replace text by 0 or 1
df2=df2.replace('GP',1)
df2=df2.replace('MS',0)
df2=df2.replace('M',1)
df2=df2.replace('F',0)
df2=df2.replace('GT3',1)
df2=df2.replace('LE3',0)
df2=df2.replace('yes',1)
df2=df2.replace('no',0)

#Data Normalisation
for i in range(0,len(df.columns)):
    df2.iloc[:,i]=(df2.iloc[:,i]-m[i])/e[i]
testsetQ2 = np.array(df2.values)

##Find the optimal alpha ?

#We use here an easy way to find a better alpha
def optialpha(trainset2,testset2,G3train,G3test):
    alpha= [0.01*l for l in range(-1000,1000)]
    alpha.remove(0)
    M = np.dot(testset2,np.dot(np.linalg.inv(np.dot(trainset2.T,trainset2) + 0.01*np.identity(trainset2.shape[1])),np.dot(trainset2.T,G3train)))
    sum=0
    for i in range(0,len(G3test)):
        sum += (M[i]-G3test[i])**2
    optRMSE=(sum/len(G3test))**(1/2)
    opta=0.01
    for val in alpha:
        X=np.dot(trainset2.T,trainset2) + val*np.identity(trainset2.shape[1])
        G3pred4 = np.dot(testset2,np.dot(np.linalg.inv(X),np.dot(trainset2.T,G3train)))
        sum=0
        for i in range(0,len(G3test)):
            sum += (G3pred4[i]-G3test[i])**2
        RM=(sum/len(G3test))**(1/2)
        if RM<=optRMSE:
            optRMSE=RM
            opta=val
    return(opta)

alpha=optialpha(trainset2,testset2,G3train,G3test)

##use of 1)e) bayesian linear model to the datas:
#Add a 1 colums to df2
testsetQ2 = np.hstack((testsetQ2,np.ones((testsetQ2.shape[0],1))))

#fonction calculing G3 in fonction of alpha:
X=np.dot(trainset2.T,trainset2) + alpha*np.identity(trainset2.shape[1])
A=np.linalg.inv(X)
w5=np.dot(A,np.dot(trainset2.T,G3train))

G3Q2 = np.dot(testsetQ2,w5)

#Write in the txt
file2=open('a08522201_1.txt','w')
v=1001
for val in G3Q2:
    file2.writelines(str(v)+"\t"+str(val)+'\n')
    v+=1
file2.close()











