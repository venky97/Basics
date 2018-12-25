import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
import csv
from sklearn.model_selection import train_test_split
raw_data = open("train.csv",'rt')
raw_data2 = open("test.csv",'rt')
read_data = list(csv.reader(raw_data,delimiter=',',quoting = csv.QUOTE_NONE))
read_data2 = list(csv.reader(raw_data2,delimiter=',',quoting = csv.QUOTE_NONE))
names = ['ID', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv']
# np.delete(read_data,0,axis=0)
# np.delete(read_data2,0,axis=0)
print(read_data)
data_train = np.array(read_data[1:]).astype('float')
data_test = np.array(read_data2[1:]).astype('float')
x = data_train[:,:-1]
y = data_train[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,random_state=1)
reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)
# regression coefficients
print('Coefficients: \n', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(x_test, y_test)))

# plot for residual error

## setting plot style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train,color="green", s=10, label='Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test,color="blue", s=10, label='Test data')

## plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

## plotting legend
plt.legend(loc='upper right')

## plot title
plt.title("Residual errors")

## function to show plot
plt.show()