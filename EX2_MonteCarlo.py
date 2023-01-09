from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata
from sklearn.model_selection import RandomizedSearchCV

# Set up the classifiers
clfs = {
    'GNB': GaussianNB(),
    'SVM': SVC(kernel='rbf'),
    # 'MLP': MLPClassifier(max_iter=5000),
    # 'DT': DecisionTreeClassifier()
}

# Set up the datasets
datasets = ['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes'
            # ,'digit', 'ecoli4', 'german', 'glass2', 'heart', 'ionosphere'
            # ,'liver', 'monkthree', 'shuttle-c0-vs-c4', 'sonar', 'soybean'
            # ,'vowel0', 'waveform', 'wisconsin', 'yeast3'
            ]

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

mean_score = np.mean(mean_scores, axis=0)
print("\nMean score:\n", mean_score)

ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
print("\nRanks:\n", ranks)

mean_ranks = np.mean(ranks, axis=0)
print("\nMean ranks:\n", mean_ranks)
