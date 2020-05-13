import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection, metrics
import warnings
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import graphviz

warnings.filterwarnings('ignore')

data = pd.read_csv('bank_data.csv', sep=';')
data[['y']] = data[['y']].replace(['no'], 0)
data[['y']] = data[['y']].replace(['yes'], 1)

# print(data['y'].value_counts())
# no     36548 , yes     4640

# ######## Do we need: Contact, Month, Day of the week ?
# ######## Discuss about 'duration'

# corr = data.corr()
# cor_plot = sns.heatmap(corr, annot=True)
# plt.show()
#y corr: 40% with 'duration', 23% with 'previous' etc

data = data.drop('poutcome', axis=1)
data = data.drop('contact', axis=1)
# data = data.drop('day_of_week', axis=1)
data = data.drop('month', axis=1)
# data = data.drop('duration', axis=1)
# data = data.drop('pdays', axis=1)

data = pd.get_dummies(data, columns=['marital', 'job', 'education'])  # , 'poutcome'
data['default'] = data['default'].map({'yes': 2, 'no': 1, 'unknown': 0})
data['housing'] = data['housing'].map({'yes': 2, 'no': 1, 'unknown': 0})
data['loan'] = data['loan'].map({'yes': 2, 'no': 1, 'unknown': 0})
# data['contact'] = data['contact'].map({'cellular': 1, 'telephone': 0})
data['day_of_week'] = data['day_of_week'].map({'fri': 4, 'thu': 3, 'wed': 2, 'tue': 1, 'mon': 0})

pd.set_option('display.expand_frame_repr', False)
print(data)

array = data.values
# X = array[:, 0:-1]
# Y = array[:, -1]

X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]
X.columns


def ml(model_in, x_input: pd.DataFrame, y_input: pd.DataFrame):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_input, y_input, test_size=0.1, random_state=5)
    model = model_in.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print("Accuracy Score: %.8f" % metrics.accuracy_score(y_test, y_predict))
    print("Precision: %.8f" % metrics.precision_score(y_test, y_predict, average="macro"))
    print("Recall: %.8f" % metrics.recall_score(y_test, y_predict, average="macro"))
    print(" F1: %.8f" % metrics.f1_score(y_test, y_predict, average="macro"))
    print(metrics.confusion_matrix(y_test, y_predict))
    # Visualize the trained tree.
    dot_data = sklearn.tree.export_graphviz(model, out_file=None,
                                            filled=True, rounded=True,
                                            special_characters=True,
                                            feature_names=x_input.columns[:])

    graph = graphviz.Source(dot_data)
    graph.render("DT")


modelDT = DecisionTreeClassifier(max_depth=4)
print("Results of Decision Trees:")
ml(modelDT, X, Y)
