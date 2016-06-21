import general_utils as Ugen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score #accuracy_score is same as model.score()

#cd ~/Documents/Programming/Python/Data\ Analyst\ ND/UD201/modules

#GET DATA
filepath='/Users/whitesi/Documents/Programming/Python/Data Analyst ND/UD201/'
df=pd.read_csv(filepath+'student-mat.csv',delimiter=';')

##CLEAN DATA
###convert all binary options to 1,0
df=df.replace(['yes','no','M','F','U','R','LE3','GT3','T','A','GP','MS']
				,[1,0,1,0,1,0,1,0,1,0,1,0])
df=df.drop(['Mjob','Fjob','reason','guardian'],axis=1) #drop categorical columns

#SET INDEPENDENT VARS
x_vars=['Medu'] ##for testing R2 scores with specific features


#INIT MODEL
Y=df['Walc'].values #or Dalc (workday alcohol consumption)
X=df[x_vars].values
# X=df.drop('Walc',axis=1) #include all vars as predictors

# logreg = linear_model.LogisticRegression()
linreg = linear_model.LinearRegression()

#SPLIT TRAIN/TEST DATA
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.1,random_state=42)

#FIT MODEL
linreg.fit(train_X, train_Y)

###CHECK MODEL RESULTS


# print(linreg.get_params())
print('Accuracy score is: %s'% linreg.score(test_X,test_Y))
print('R2 score is: %s'% r2_score(train_Y,linreg.predict(train_X)))
print('Coefficients are: %s'% linreg.coef_)
# print(linreg.predict([120000,2,2,1,21]))



#######NOTES TO SELF#############
##





