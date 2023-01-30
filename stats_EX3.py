from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import VotingClassifier
from scipy.stats import ranksums
from tabulate import tabulate

# Set up the classifiers
clfs = {
    'GNB': GaussianNB(),
    'SVM': SVC(kernel='rbf'),
    'MLP': MLPClassifier(max_iter=5000),
    'DT': DecisionTreeClassifier(),
    'Bag': VotingClassifier(estimators='drop')
}

# Set up the datasets
datasets = ['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes',
            'digit', 'ecoli4', 'german', 'glass2', 'heart'
            # 'ionosphere', 'liver', 'monkthree', 'shuttle-c0-vs-c4', 'sonar', 'soybean'
            # , 'vowel0', 'waveform', 'wisconsin', 'yeast3'
            ]

#scores = np.load('results_ex3G.npy')
scores = np.load('results_ex3M.npy')

headers = list(clfs.keys())
names_column = np.expand_dims(np.array(datasets), axis=1)

mean_scores = np.mean(scores, axis=2).T
mean_scores_table = np.concatenate((names_column, mean_scores), axis=1)
mean_scores_table = tabulate(mean_scores_table, headers, floatfmt=".3f")
print("\nMean scores:\n", mean_scores_table)

ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
ranks_table = np.concatenate((names_column, ranks), axis=1)
ranks_table = tabulate(ranks_table, headers, floatfmt=".1f")
print("\nRanks:\n", ranks_table)

mean_ranks = np.mean(ranks, axis=0)
b = []
for i in range(len(mean_ranks)):
    b.append('')
mean_ranks_table = []
for i in range(len(mean_ranks)):
    mean_ranks_table.append([mean_ranks[i], b[i]])
mean_ranks_table = tabulate(np.array(mean_ranks_table).T, headers, floatfmt=".1f")
print("\nMean ranks:\n", mean_ranks_table)

'''============================================================================================'''

alfa = .05
w_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))
'''=== test Wilcoxona ==='''
for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

headers = list(clfs.keys())
names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)
