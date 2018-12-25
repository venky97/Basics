import numpy as np
import matplotlib.pyplot as plt

x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([1,3,2,5,7,8,8,9,10,12])
p0 = np.array([1,1])
n = np.size(x)
# Least squares technique
m_x = np.mean(x)
m_y = np.mean(y)
ss_xy = np.sum(x*y-n*m_x*m_y)
ss_xx = np.sum(x*x - n*m_x*m_x)
p0[0] = ss_xy/ss_xx
plt.scatter(x,y,color='m',marker='o',s=30)
y_pred = p0[0] + x*p0[1]
plt.plot(x,y_pred,color = 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.show()