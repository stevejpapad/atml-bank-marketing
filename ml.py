import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn import model_selection, metrics
import graphviz

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def trees(x_input: pd.DataFrame, y_input: pd.DataFrame, cost_weight):
    model = DecisionTreeClassifier(max_depth=4, class_weight=cost_weight)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_input, y_input,
                                                                        test_size=0.1)  # random_state=5
    model = model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    evaluate(y_test, y_predict, cost_weight)

    # Visualize the trained tree.
    dot_data = sklearn.tree.export_graphviz(model, out_file=None,
                                            filled=True, rounded=True,
                                            special_characters=True,
                                            feature_names=x_input.columns[:])

    graph = graphviz.Source(dot_data)
    graph.render("DT")


def evaluate(y_test, y_predict, cost_weight):

    print("Accuracy Score: %.4f" % metrics.accuracy_score(y_test, y_predict))
    print("Precision: %.4f" % metrics.precision_score(y_test, y_predict, average="macro"))
    print("Recall: %.4f" % metrics.recall_score(y_test, y_predict, average="macro"))
    print(" F1: %.4f" % metrics.f1_score(y_test, y_predict, average="macro"))
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_predict).ravel()
    print(metrics.confusion_matrix(y_test, y_predict))
    total_cost = cost_weight[1]*fn + cost_weight[0]*fp
    print("Cost: %.f" % total_cost)