from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from scipy.stats import rankdata
from tabulate import tabulate
from scipy.stats import ranksums

# Set up the classifiers
clfs = {
    'GNB': GaussianNB(),
    'SVM': SVC(kernel='rbf'),
    'MLP': MLPClassifier(max_iter=5000),
    'DT': DecisionTreeClassifier()
}

# Set up the datasets
datasets = ['GridSearch', 'MonteCarlo']

scores1 = np.load('results_ex1G.npy')
scores2 = np.load('results_ex1M.npy')
scores = np.concatenate((scores1, scores2), axis=1)

headers = datasets
names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)

mean_scores = np.mean(scores, axis=2)
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
w_statistic = np.zeros((len(datasets), len(datasets)))
p_value = np.zeros((len(datasets), len(datasets)))
'''=== test Wilcoxona ==='''
for i in range(len(datasets)):
    for j in range(len(datasets)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

headers = datasets
names_column = np.expand_dims(np.array(datasets), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(datasets), len(datasets)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((len(datasets), len(datasets)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)
