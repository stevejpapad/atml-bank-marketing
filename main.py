import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from ml import trees, keras_nn, random_trees
from data_prep import prep, min_max_scale
from imbalance_handling import handle_imbalance
from explainability import tree_viz, tree_to_text, tree_feature_importance, tree_bar_interpretation

data = pd.read_csv('bank_data.csv', sep=';')
data = prep(data)
# data = min_max_scale(data)

X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]
# X_res, Y_res = handle_imbalance(X, Y, under=True, over=False, combine=False)
X_res, Y_res = handle_imbalance(X, Y, under=False, over=False, combine=True)
# X_res, Y_res = handle_imbalance(X, Y, under=False, over=False, combine=True)
cost_weight = {0: 1, 1: 10}

# x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1, random_state=5)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X_res, Y_res, test_size=0.1, random_state=5)

print("--- Decision Trees ---")
# tree = trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=False)
# tree = trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=True)
# tree = trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=False)
# tree = trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=True)

# print("--- Random Forests ---")
model = random_trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=False)
# model = random_trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=True)
# model = random_trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=False)
# model = random_trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=True)

# Explainability section
# White box models
# tree_viz(tree, X)
# tree_to_text(tree, list(X.columns))
# tree_feature_importance(tree, list(X.columns))
# tree_bar_interpretation(tree, X)

# Black box models surrogate
new_y_train = model.predict(x_train)
tree = trees(x_train, x_test, new_y_train, y_test, cost_weight, sensitive=False)
# tree_viz(tree, X)
# tree_to_text(tree, list(X.columns))
# tree_feature_importance(tree, list(X.columns))
tree_bar_interpretation(tree, X)
