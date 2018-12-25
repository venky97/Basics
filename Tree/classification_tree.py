'''
Decision tree is a type of supervised learning algorithm (having a pre-defined target variable) that is mostly used in
classification problems. It works for both categorical and continuous input and output variables. In this technique, we
 split the population or sample into two or more homogeneous sets (or sub-populations) based on most significant
 splitter / differentiator in input variables.
 Regression Trees vs Classification Trees
 Gini Index
'''
#Import Library
from sklearn import datasets, linear_model, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import csv
import numpy as np#use RandomForestRegressor for regression problem
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Random Forest object
def load_csv(filename):
    rd = open(filename,"rt")
    data = np.array(list(csv.reader(rd,delimiter=',',quoting = csv.QUOTE_NONE)))
    return data
# data = load_csv("dataset1.csv")
# X = data[:,:-1]
# y = data[:,-1]
digits = datasets.load_digits()# load the digit dataset
X = digits.data# defining feature matrix(X) and response vector(y)
y = digits.target
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=1)
model= RandomForestClassifier(n_estimators=1000)
# Train the model using the training sets and check score
model.fit(x_train, y_train)
#Predict Output
predicted= model.predict(x_test)
print(metrics.accuracy_score(y_test,predicted)*100)

