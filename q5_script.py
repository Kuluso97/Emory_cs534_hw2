import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

## Load Data
df = pd.read_csv('hw2_data_2.txt', sep='\t')
X_train, y_train = np.array(df.iloc[:700, :-1]), np.array(df.iloc[:700, -1])
X_test, y_test = np.array(df.iloc[700:, :-1]), np.array(df.iloc[700:, -1])

clf = GradientBoostingClassifier(loss='deviance')
clf.fit(X_train, y_train)
print("The testing error rate for the gradient boosting classifier is %.4f" 
										%(1-clf.score(X_test,y_test)))

## Rank the variables by importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = df.columns[:-1]

for f in range(X_train.shape[1]):
	print ("%2d) %-*s %f" %(f + 1, 30,
							feat_labels[indices[f]],
							importances[indices[f]]))





