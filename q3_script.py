import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('hw2_data_2.txt', sep='\t')
X_train, y_train = np.array(df.iloc[:700, :-1]), np.array(df.iloc[:700, -1])
X_test, y_test = np.array(df.iloc[700:, :-1]), np.array(df.iloc[700:, -1])

svm = SVC()
gamma_range = [1 * (10 ** i) for i in range(-4,4)]

def grid_search(grid, param_):

	### grid search using 10 fold cross-validation
	gs = GridSearchCV(estimator=svm,
				param_grid=grid,
				scoring='accuracy',
				cv=10,
				n_jobs=-1)

	gs = gs.fit(X_train, y_train)
	mean_scores = gs.cv_results_['mean_test_score']
	error_rates = 1 - mean_scores
	best_param_ = gs.best_params_[param_]

	return error_rates, best_param_

def plot_result(param, error_rates, kernel, xlabels):
	length = len(error_rates)
	fig, ax = plt.subplots()
	x = list(range(length))
	my_xticks = list(map(str, xlabels))
	ax.set_xticks(x)
	ax.set_xticklabels(my_xticks)

	ax.plot(x, error_rates)
	ax.set(xlabel=param, ylabel='cross-validation error rate',
       		title='{} v. Cross Validation Error Rate ({})'.format(param, kernel))

## Radial Kernel

grid_rbf = [{'kernel':['rbf'], 'gamma': gamma_range}]
error_rates, best_gamma = grid_search(grid_rbf, 'gamma')
print("The best gamma for the radial kernel is %s" %best_gamma)
svm_r = SVC(kernel='rbf', gamma=best_gamma)
svm_r.fit(X_train, y_train)
print("The best testing error rate for the radial kernel SVM is %.4f" %(1-svm_r.score(X_test, y_test)))

plot_result('gamma', error_rates, 'Radial', gamma_range)

# plt.show()

## Sigmoid Kernel

grid_sig = [{'kernel':['sigmoid'], 'gamma': gamma_range}]
error_rates, best_gamma = grid_search(grid_sig, 'gamma')
print("The best gamma for the sigmoid kernel is %s" %best_gamma)
svm_s = SVC(kernel='sigmoid', gamma=best_gamma)
svm_s.fit(X_train, y_train)
print("The best testing error rate for the sigmoid kernel SVM is %.4f" %(1-svm_s.score(X_test, y_test)))

plot_result('gamma', error_rates, 'Sigmoid', gamma_range)

# plt.show()

## Polynomial Kernel
degree_range = list(range(1,11))
grid_poly = [{'kernel':['poly'], 'degree': degree_range}]
error_rates, best_degree = grid_search(grid_poly, 'degree')
print("The best degree for the polynomial kernel is %s" % best_degree)
svm_p=SVC(kernel='poly', degree=best_degree)
svm_p.fit(X_train, y_train)
print("The best testing error rate for the polynomial kernel SVM is %.4f" %(1-svm_p.score(X_test, y_test)))

plot_result('degree', error_rates, 'Polynomial', degree_range)

plt.tight_layout()
plt.show()





