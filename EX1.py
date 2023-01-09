from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import rankdata
import csv
import os
from sklearn.model_selection import RandomizedSearchCV

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=42)

# create the folder if it doesn't exist
if not os.path.exists('datasets'):
    os.makedirs('datasets')

# open the file in write mode
with open('datasets/synth_dataset.csv', 'w', newline='') as f:
    # create a CSV writer
    writer = csv.writer(f)

    # write the data rows
    for x, y in zip(X, y):
        writer.writerow(list(x) + [y])

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the classifiers
clfs = {
    'GNB': GaussianNB(),
    'SVM': SVC(kernel='rbf'),
    # 'MLP': MLPClassifier(max_iter=5000),
    # 'DT': DecisionTreeClassifier()
}

datasets = {
    'synth_dataset'
}

# Set up the hyperparameter grids
param_grid = [{'var_smoothing': [1e-9, 1e-3]},
              {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}
              # ,{'hidden_layer_sizes': [(10,), (20,), (30,), (40,)], 'alpha': [0.1, 0.01, 0.001, 0.0001]}
              # ,{'max_depth': [2, 4, 6, 8, 10], 'min_samples_split': [2, 4, 6, 8, 10]
              #  ,'min_samples_leaf': [1, 2, 4, 6, 8]}
              ]
# ==================================================================================================

n_datasets = len(datasets)
n_splits = 2  # 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % dataset, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        print(len(X),len(train),len(test))
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])

            # Set up the grid search
            grid_search = GridSearchCV(clf, param_grid[clf_id], cv=5, verbose=1, n_jobs=-1)
            # Fit the grid search to the data
            grid_search.fit(X[train], y[train])

            # Get the best hyperparameters
            best_params = grid_search.best_params_
            # Get the best estimator
            best_estimator = grid_search.best_estimator_

            # Predict on the validation set using the GNB classifier
            y_pred = best_estimator.predict(X[test])
            # Calculate the accuracy of the GNB classifier
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)

scores = np.load('results.npy')
print("\nScores:\n", scores.shape)

mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)

print("MonteCarlo ========================================")

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % dataset, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])

            # Calculate max number of combinations in param_grid
            max_iter = 1
            for key in param_grid[clf_id]:
                max_iter *= len(param_grid[clf_id][key])

            # create the randomized search object
            search = RandomizedSearchCV(clf, param_grid[clf_id], cv=5, n_iter=max(max_iter / 3, 1), random_state=0)

            # fit the search object to the training data
            search.fit(X[train], y[train])

            # Get the best hyperparameters
            best_params = search.best_params_
            # Get the best estimator
            best_estimator = search.best_estimator_

            # Predict on the validation set using the GNB classifier
            y_pred = best_estimator.predict(X[test])
            # Calculate the accuracy of the GNB classifier
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)

scores = np.load('results.npy')
print("\nScores:\n", scores.shape)

mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)
