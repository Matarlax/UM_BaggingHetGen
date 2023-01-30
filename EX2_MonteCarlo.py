from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm

# Set up the classifiers
clfs = {
    'GNB': GaussianNB(),
    'SVM': SVC(kernel='rbf'),
    'MLP': MLPClassifier(max_iter=5000),
    'DT': DecisionTreeClassifier()
}

# Set up the datasets
datasets = ['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes',
            'digit', 'ecoli4', 'german', 'glass2', 'heart'
            # 'ionosphere', 'liver', 'monkthree', 'shuttle-c0-vs-c4', 'sonar', 'soybean'
            # , 'vowel0', 'waveform', 'wisconsin', 'yeast3'
            ]

# Set up the hyperparameter grids
param_grid = [{'var_smoothing': [1e-9, 1e-3, 1e-6, 1e-12]},
              {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]},
              {'hidden_layer_sizes': [(10,), (20,), (30,)], 'alpha': [0.1, 0.01, 0.001]},
              {'max_depth': [2, 4, 6, 8], 'min_samples_split': [2, 4, 6, 8],
               'min_samples_leaf': [1, 2, 4, 6]}
              ]
# ==================================================================================================

n_datasets = len(datasets)
n_splits = 2  # 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

for data_id, dataset in tqdm(enumerate(datasets), total=len(datasets), desc="Dataset", colour="blue"):
    dataset = np.genfromtxt("datasets/%s.csv" % dataset, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in tqdm(enumerate(rskf.split(X, y)), total=n_splits * n_repeats, desc="Fold"):
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
            #print(best_params)
            # Get the best estimator
            best_estimator = search.best_estimator_

            # Predict on the validation set using the GNB classifier
            y_pred = best_estimator.predict(X[test])
            # Calculate the accuracy of the GNB classifier
            scores[clf_id, data_id, fold_id] = balanced_accuracy_score(y[test], y_pred)

np.save('results_ex2M', scores)
