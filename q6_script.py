import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pyearth import Earth

## Load Data
df = pd.read_csv('hw2_data_2.txt', sep='\t')
X_train, y_train = np.array(df.iloc[:700, :-1]), np.array(df.iloc[:700, -1])
X_test, y_test = np.array(df.iloc[700:, :-1]), np.array(df.iloc[700:, -1])

## Using py-earth package, please install it follow the instructions in README file
clf = Earth()
clf.fit(X_train, y_train)

## Predict the value and calculate the testing error rate
pred_vals = clf.predict(X_test)

## Dichotomize the predicted outcome at the median
median = np.median(pred_vals)
res = np.where(pred_vals >= median, 1, -1)
# print(res)

error_rate = 1 - sum(res == y_test)/y_test.shape[0]
print("The testing error rate for MARS classifier is: %.4f" %error_rate)
