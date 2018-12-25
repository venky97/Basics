from sklearn import datasets, linear_model, metrics
import numpy as np
digits = datasets.load_digits()# load the digit dataset
X = digits.data# defining feature matrix(X) and response vector(y)
y = digits.target
from sklearn.model_selection import train_test_split# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=10)
reg = linear_model.LogisticRegression()# create logistic regression object
reg.fit(X_train, y_train)# train the model using the training sets
y_pred = reg.predict(X_test)# making predictions on the testing set
print("Logistic Regression model accuracy(in %):",metrics.accuracy_score(y_test, y_pred)*100)# comparing actual response values (y_test) with predicted response values (y_pred)
