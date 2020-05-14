import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from ml import trees, keras_nn, random_trees
from data_prep import prep, min_max_scale
from imbalance_handling import handle_imbalance

data = pd.read_csv('bank_data.csv', sep=';')
data = prep(data)
# data = min_max_scale(data)

X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]
X_res, Y_res = handle_imbalance(X, Y, under=True, over=False, combine=False)
cost_weight = {0: 1, 1: 10}

print("--- Decision Trees ---")
trees(X, Y, cost_weight, sensitive=False)
trees(X, Y, cost_weight, sensitive=True)
trees(X_res, Y_res, cost_weight, sensitive=False)
trees(X_res, Y_res, cost_weight, sensitive=True)

# print("--- Random Forests ---")
# random_trees(X, Y, cost_weight, sensitive=False)
# random_trees(X, Y, cost_weight, sensitive=True)
# random_trees(X_res, Y_res, cost_weight, sensitive=False)
# random_trees(X_res, Y_res, cost_weight, sensitive=True)
