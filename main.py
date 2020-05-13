import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from ml import trees, sv_class, keras_nn
from data_prep import prep, min_max_scale

data = pd.read_csv('bank_data.csv', sep=';')
data = prep(data)
pd.set_option('display.expand_frame_repr', False)
print(data)
# data = min_max_scale(data)

X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]

cost_weight = {0: 1, 1: 10}

# print("Decision Trees with CS:")
# trees(X, Y, cost_weight)
# print("\n")
# print("SVM with CS:")
# sv_class(X, Y, cost_weight)
print("NN:")
keras_nn(X, Y, cost_weight)

