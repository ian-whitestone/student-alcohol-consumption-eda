import general_utils as Ugen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score #accuracy_score is same as model.score()
from scipy import stats
from sklearn.feature_selection import f_regression


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

# print (list(df.columns.values))

# ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
#	'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 
#	'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 
#    'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']


x_vars=['Medu'] ##for testing R2 scores with specific features


#INIT MODEL
Y=df['Walc'].values #or Dalc (workday alcohol consumption)
# X=df[x_vars].values
X=df.drop(['Walc','Dalc'],axis=1) #include all vars as predictors
x_vars=list(df.drop(['Walc','Dalc'],axis=1).columns.values) 
# linreg = linear_model.LinearRegression()

##http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html
##Ian: read baout f-score etc...
##http://blog.minitab.com/blog/adventures-in-statistics/how-to-interpret-regression-analysis-results-p-values-and-coefficients
F,p_val=f_regression(X,Y)

##p-values greater > 0.05 can be removed - Read about other methods, refresher on why this is the way


good_vars={var_name:p for var_name,p in zip(x_vars,p_val) if p<0.05}
bad_vars={var_name:p for var_name,p in zip(x_vars,p_val) if p>0.05}

print (good_vars.keys())

#dict_keys(['studytime', 'sex', 'G1', 'famsize', 'famrel', 'freetime', 'failures', 
	#'higher', 'goout', 'traveltime', 'age', 'address', 'nursery', 'absences'])

print (bad_vars.keys())
# dict_keys(['school', 'Fedu', 'romantic', 'Medu', 'Pstatus', 'activities', 'famsup',
	# 'G3', 'G2', 'internet', 'paid', 'health', 'schoolsup'])


# #SPLIT TRAIN/TEST DATA
# train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.001,random_state=42) #currently don't need any test data

# #FIT MODEL
# linreg.fit(train_X, train_Y)

# ###CHECK MODEL RESULTS

# print('Accuracy score is: %s'% linreg.score(test_X,test_Y))
# print('R2 score is: %s'% r2_score(train_Y,linreg.predict(train_X)))
# print('Coefficients are: %s'% linreg.coef_)



#DATA PLOTTING
# x = [x[0] for x in df[['goout']].values] #definitely a better way to do this but losing patience
# plt.plot(x,Y,'o')

# # calc the trendline
# z = np.polyfit(x, Y, 1)
# p = np.poly1d(z)
# plt.plot(x,p(x),"r--")

# plt.show()

# #######NOTES TO SELF#############
# ##add CHI-sQUARED test etc.





