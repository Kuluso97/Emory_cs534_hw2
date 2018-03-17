import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('hw2_data_2.txt', sep='\t')
X_train, y_train = np.array(df.iloc[:700, :-1]), np.array(df.iloc[:700, -1])
X_test, y_test = np.array(df.iloc[700:, :-1]), np.array(df.iloc[700:, -1])

oob_scores = []
trees = list(range(10, 510, 10))

for i in trees:
	forest_clf = RandomForestClassifier(n_estimators=i, oob_score=True)
	forest_clf.fit(X_train, y_train)
	oob_scores.append(1-forest_clf.oob_score_)
	print("The OOB error rate for %s trees is: %.4f" %(i, 1-forest_clf.oob_score_))

## plot oob error rate v.s. the number of estimators
fig, ax = plt.subplots()
ax.plot(trees, oob_scores)
ax.set(xlabel='the number of trees', ylabel='OOB error rate',
   		title='OOB error rate v.s. the number of trees')

plt.show()

## The oob_score of 500 trees
target = oob_scores[-1]
index = -1

for i in range(len(oob_scores)):
	if oob_scores[i] <= target + 0.002:
		index = i
		break

## Fit the model 
num_trees = (index + 1) * 10
clf = RandomForestClassifier(n_estimators=num_trees)
clf.fit(X_train, y_train)

print("The error rate for the random forest classifier with %s estimators is %.4f" %(num_trees, 1-clf.score(X_test,y_test)))

## Rank the variables by importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = df.columns[:-1]

for f in range(X_train.shape[1]):
	print ("%2d) %-*s %f" %(f + 1, 30,
							feat_labels[indices[f]],
							importances[indices[f]]))