# https://www.kaggle.com/ajay1735/my-credit-scoring-model

# Create a credit scoring algorithm that predicts the chance of a given loan applicant defaulting on loan repayment.

# importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# reading the input

df=pd.read_csv(r'/home/ram/Downloads/kaggle/hmeq.csv')

# glimpse of the dataset
print(df.head())
print(df.info())
print(df.columns)
l=list(df.columns)


# Imputing the input variables
print(df.isnull().sum())
for i in range(len(l)):
    print(df[l[i]].value_counts())



# Nominal features
# Replacement using majority class
# majority class in case of JOB variable is Other
# majority class in case of REASON varibale is DebtCon

df["REASON"].fillna(value = "DebtCon",inplace = True)
df["JOB"].fillna(value = "Other",inplace = True)

df["DEROG"].fillna(value=0,inplace=True)
df["DELINQ"].fillna(value=0,inplace=True)
# Numeric features
# Replacement using mean of each class

df.fillna(value=df.mean(),inplace=True)

print(df.isnull().sum())

# Applying the models on the data after imputation
# Applying the basic Classification on the data after replacement/imputation.
# Lets check the performnace by applying both Logistic Regression and Decision tree algorithms.
# Before applying the algorithms,
# The data is split into training and testing sets in the ratio 2:1 that is test data 33% and train data 67%.
# And also taking all the columns except JOB,REASON as input features(as they are nominal features,
# they must be transformed to other variables to be usable which is taken care of in next section).

# removing the features BAD,JOB,REASON from the input features set
x_basic = df.drop(columns=["BAD","JOB","REASON"])
y = df["BAD"]

# Spliting the data into test and train sets
x_basic_tr,x_basic_te,y_tr,y_te = train_test_split(x_basic,y,test_size =.33,random_state=1)
logreg_basic = LogisticRegression()
print("---------------logistic regression----------------")
# Training the basic logistic regression model with training set
logreg_basic.fit(x_basic_tr,y_tr)

# Predicting the output of the test cases using the algorithm created above
y_pre = logreg_basic.predict(x_basic_te)
accuracy = logreg_basic.score(x_basic_te, y_te)
print("log rig:Accuracy on test data for a given model is {}".format(accuracy))
print("confussion matrix")
# confussion matrix
print(confusion_matrix(y_te,y_pre))
# classification report
print("classification report")

print(classification_report(y_te,y_pre))
print("-----------------decision tree----------------")
dectree_basic = DecisionTreeClassifier()
dectree_basic.max_depth = 100
# Training the basic Decision Tree model with training set
dectree_basic.fit(x_basic_tr,y_tr)
y_pre = dectree_basic.predict(x_basic_te)
accuracy = dectree_basic.score(x_basic_te, y_te)
print("decision tree:Accuracy on test data for a given model is {}".format(accuracy))
# confussion matrix
print("confussion matrix")

print(confusion_matrix(y_te,y_pre))
# classification report
print("classification report")

from sklearn.ensemble import GradientBoostingClassifier
print(classification_report(y_te,y_pre))
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(x_basic_tr, y_tr)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(x_basic_tr, y_tr)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(x_basic_te, y_te)))
    print()