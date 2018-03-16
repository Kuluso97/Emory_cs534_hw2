import numpy as np 
import pandas as pd
import copy 

class Adaboost(object):

	def __init__(self, base_clf, iters):
		self.base_clf = base_clf
		self.iters = iters

	def fit(self, X_train, y_train):
		self.weights = np.ones(X_train.shape[0]) / X_train.shape[0]
		self.alphas = []
		self.weak_learners = []

		for i in range(self.iters):
			learner = copy.deepcopy(self.base_clf.fit(X_train, y_train, sample_weight=self.weights))
			self.weak_learners.append(learner)

			y_pred = self.base_clf.predict(X_train)
			err = np.dot(self.weights, y_pred != y_train) / np.sum(self.weights)
			alpha_m = np.log2((1-err)/float(err))
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


## Simple implementation(only can be used in this setting)
class DecisionTreeStump(object):

	def __init__(self):
		self.criterion = None

	def fit(self, X_train, y_train, sample_weight):
		self.X_train, self.y_train = X_train, y_train
		x1_s = np.sort(np.unique(X_train[:, 0]))
		x2_s = np.sort(np.unique(X_train[:, 1]))
		entropy_min = float('inf')
		res = (x1_s[0], 0)

		for sep in x1_s[:-1]:
			left = list(zip(y_train[X_train[:,0] <= sep], sample_weight[X_train[:,0] <= sep]))
			right = list(zip(y_train[X_train[:,0] > sep], sample_weight[X_train[:,0] > sep]))
			entropy = self.__entropy(left) + self.__entropy(right)
			if entropy < entropy_min:
				entropy_min = entropy
				res = (sep, 0)

		for sep in x2_s[:-1]:
			left = list(zip(y_train[X_train[:, 1] <= sep], sample_weight[X_train[:, 1] <= sep]))
			right = list(zip(y_train[X_train[:, 1] > sep], sample_weight[X_train[:, 1] > sep]))
			entropy = self.__entropy(left) + self.__entropy(right)
			if entropy < entropy_min:
				entropy_min = entropy
				res = (sep, 1)

		self.criterion = res

		return self

	def __entropy(self, points):
		group_1 = [point for point in points if point[0] == 1]
		group_2 = [point for point in points if point[0] == -1]
		weight_1, weight_2 = 0.0, 0.0
		for point in group_1:
			weight_1 += point[1]

		for point in group_2:
			weight_2 += point[1]

		p1 = weight_1 / (weight_1 + weight_2)
		p2 = 1 - p1

		return -sum([pi * np.log2(pi) for pi in (p1,p2) if pi > 0])

	def predict(self, X_test):
		axis = self.criterion[1]
		label = 1 if np.sum(self.y_train[self.X_train[:, axis] <= self.criterion[0]]) > 0 else -1
		return np.where(X_test[:, axis] <= self.criterion[0], label, -label)
		

def main():
	## Import data 
	df = pd.read_csv('hw2_data_1.txt', sep='\t')

	X_train, y_train = np.array(df.iloc[:70,:-1]), np.array(df.iloc[:70,-1])
	X_test, y_test = np.array(df.iloc[70:,:-1]), np.array(df.iloc[70:,-1])

	## Create base learner: 1-depth decision tree classifier (decision stump)
	tree_clf = DecisionTreeStump()

	for iter_ in (3,5,10,20):
		ada = Adaboost(tree_clf, iter_)
		ada.fit(X_train, y_train)
		res = ada.predict(X_test)
		print("The error rate of the Adaboost classifier after %s iterations is %.2f" 
									%(iter_, (1-ada.score(X_test,y_test))))



if __name__ == '__main__':
	main()

