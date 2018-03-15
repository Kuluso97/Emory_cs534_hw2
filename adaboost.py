import numpy as np 
import pandas as pd
import copy 
from sklearn.tree import DecisionTreeClassifier

class Adaboost(object):

	def __init__(self, base_clf, iters):
		self.base_clf = base_clf
		self.iters = iters

	def fit(self, X_train, y_train):
		self.weights = np.ones(X_train.shape[0]) / X_train.shape[0]
		self.alphas = []
		self.weak_learners = []

		for i in range(self.iters):
			learner = copy.deepcopy(self.base_clf.fit(X_train, y_train, sample_weight = self.weights))
			self.weak_learners.append(learner)

			y_pred = self.base_clf.predict(X_train)
			err = np.dot(self.weights, y_pred != y_train) / np.sum(self.weights)
			alpha_m = np.log((1-err)/float(err))
			self.alphas.append(alpha_m)

			self.weights = np.multiply(self.weights, np.exp(alpha_m * (y_pred != y_train)))

	def predict(self, X_test):
		res = np.zeros(X_test.shape[0])
		for i in range(self.iters):
			res += self.alphas[i] * self.weak_learners[i].predict(X_test)

		return np.where(res >= 0.0, 1, -1)

	def score(self, X_test, y_test):
		res = self.predict(X_test)
		return sum(res==y_test)/y_test.shape[0]

def main():
	## Import data 
	df = pd.read_csv('hw2_data_1.txt', sep='\t')

	X_train, y_train = np.array(df.iloc[:70,:-1]), np.array(df.iloc[:70,-1])
	X_test, y_test = np.array(df.iloc[70:,:-1]), np.array(df.iloc[70:,-1])

	## Create base learner: 1-depth decision tree classifier (decision stump)
	tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
	
	for iter_ in (3,5,10,20):
		ada = Adaboost(tree_clf, iter_)
		ada.fit(X_train, y_train)
		res = ada.predict(X_test)
		print("The error rate of the Adaboost classifier after %s iterations is %.2f" 
									%(iter_, (1-ada.score(X_test,y_test))))



if __name__ == '__main__':
	main()

