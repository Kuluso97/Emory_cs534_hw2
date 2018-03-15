import numpy as np
import pandas as pd


## Simple implementation of Perceptron

class Perceptron(object):

	def __init__(self, learning_rate = 1, iterations = 50):
		self.learning_rate = learning_rate
		self.iterations = iterations;

	def fit(self, X, y):
		## Initial all weights to 1
		self.weights = np.ones(1 + X.shape[1])

		for _ in range(self.iterations):

			for xi, target in zip(X, y):
				diff = self.learning_rate * (target - self.predict(xi))
				self.weights[1:] += diff * xi
				self.weights[0] += diff

		return self

	def predict(self, X):
		v_sum = np.dot(X, self.weights[1:]) + self.weights[0]
		return np.where(v_sum >= 0.0, 1, -1)

	def score(self, X_test, y_test):
		res = self.predict(X_test)
		return sum(res==y_test)/y_test.shape[0]

def main():

	## Import data 

	df = pd.read_csv('hw2_data_1.txt', sep='\t')

	X_train, y_train = np.array(df.iloc[:70,:-1]), np.array(df.iloc[:70,-1])
	X_test, y_test = np.array(df.iloc[70:,:-1]), np.array(df.iloc[70:,-1])

	## Train data

	clf = Perceptron(learning_rate=1, iterations=50)
	clf.fit(X_train, y_train)

	print("The error rate of the Perceptron classifier after 50 iterations is %.2f" %(1-clf.score(X_test,y_test)))

if __name__ == '__main__':
	main()