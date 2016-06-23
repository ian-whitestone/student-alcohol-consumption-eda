import general_utils as Ugen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score #accuracy_score is same as model.score()

#cd ~/Documents/Programming/Python/Data\ Analyst\ ND/UD201/modules

#GET DATA
df=pd.read_csv('../default of credit card clients.csv',skiprows=[0])

#SET INDEPENDENT VARS
x_vars=['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE']
# x_vars=['PAY_2']


#INIT MODEL
Y=df['default payment next month'].values
# X=df[x_vars].values
X=df.drop('default payment next month',axis=1) #include all vars as predictors

logreg = linear_model.LogisticRegression()

#SPLIT TRAIN/TEST DATA
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.1,random_state=42)

#FIT MODEL
logreg.fit(train_X, train_Y)

###CHECK MODEL RESULTS


# print(logreg.get_params())
print('Accuracy score is: %s'% logreg.score(test_X,test_Y))
print('R2 score is: %s'% r2_score(train_Y,logreg.predict(train_X)))
print('Coefficients are: %s'% logreg.coef_)
# print(logreg.predict([120000,2,2,1,21]))



								#######NOTES TO SELF#############
##currently getting accruacy score of ~70% no matter what variables are added
##looking at data, 80% of Y's are 0, so if you just guessed 0 you'd be right about 80% of the time..
##in other words, we have no improvement over the mean...
######DIAGNOSIS: imbalanced dataset....apparently a common problem...research methods for dealing with this!
# http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/


##update: I am getting negative R2 scores with the given linear, logistic model
##this means that the model performs worse than a horizontal line






