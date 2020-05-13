import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


from ml import trees
from data_prep import prep, min_max_scale, handle_imbalance

data = pd.read_csv('bank_data.csv', sep=';')
data = prep(data)

# data = min_max_scale(data)

X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]
X_res, Y_res = handle_imbalance(X,Y,under = False,over=True,ensemble = False)
corr = data.corr()
cor_plot = sns.heatmap(corr, annot=True)
plt.show()
# NOTE: y corr: 40% with 'duration', 23% with 'previous' etc

cost_weight = {0: 1, 1: 10}
print("Results of Decision Trees:")
trees(X, Y, cost_weight)
