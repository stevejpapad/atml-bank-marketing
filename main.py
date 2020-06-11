import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from cost_sensitive_ml import exp_trees, exp_random_forests, exp_ada, exp_cost_tree, trees
from data_prep import prep, min_max_scale
from imbalance_handling import handle_imbalance
from cost_sensitive_ml import exp_imbalance
# from explainability import tree_viz, tree_to_text, tree_feature_importance, tree_bar_interpretation, lime_local, \
#     tree_dependency_plot, tree_local_interpretation

data = pd.read_csv('bank_data.csv', sep=';')
data = prep(data)
# data = min_max_scale(data)

X = data.drop(['y'], axis=1)
Y = data['y']

cost_weight = {0: 1, 1: 10}
exp_trees(X, Y, cost_weight)
exp_random_forests(X, Y, cost_weight)
exp_ada(X, Y, cost_weight)
exp_cost_tree(X, Y, cost_weight)

exp_imbalance(X, Y,cost_weight)

# tree, x_train, x_test, y_test = exp_trees(X, Y, cost_weight)
# model, x_train, x_test, y_test = exp_random_forests(X, Y, cost_weight)
# model, x_train, x_test, y_test = exp_ada(X, Y, cost_weight)
# tree, x_train, x_test, y_test = exp_cost_tree(X, Y, cost_weight)
#
#
# # Explainability section
# print("--------------Explainability section--------------")
# # new_y_train = pd.Series(model.predict(x_train))
# # tree = trees(x_train, x_test, new_y_train, y_test, cost_weight, sensitive=False)
# tree_viz(tree, X)
# print("Rules")
# tree_to_text(tree, list(X.columns))
# # tree_feature_importance(tree, list(X.columns))
# tree_bar_interpretation(tree, X)
# y_pred = tree.predict(x_train)
# print("Tree local interpretation")
# tree_local_interpretation(tree, x_test, random.randint(0, len(x_test)), y_pred)
# dataset = pd.concat([X, Y], axis=1)
# # tree = model
# tree_dependency_plot(tree, dataset)
# lime_local(tree, x_train, x_test, y_test)
