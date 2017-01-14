import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# read data
dataframe = pd.read_csv('db.txt')
x_values = dataframe[['x']]
y_values = dataframe[['y']]

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values,y_values)

plt.scatter(x_values,y_values)
plt.plot(x_values, body_reg.predict(x_values))

# calculate propabilistics characteristics
error = body_reg.predict(x_values.values) - y_values.values	# error every existing point
error_mean = np.mean(error) # mean error
error_std = np.std(error) # standard deviation

print('Error mean: {:.2f}, STD: {:.2f}'.format(error_mean, error_std))
plt.show()

# error histogram
n, bins, patches = plt.hist(error, normed=1, facecolor='blue', alpha = 0.75)

# plotting error
y = mlab.normpdf(bins,error_mean,error_std)
l = plt.plot(bins,y ,'r--',linewidth=1)

plt.xlabel('Error intervals')
plt.ylabel('Error intervals probability')
plt.show()