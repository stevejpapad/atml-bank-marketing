import pandas as pd
import sklearn
import numpy as np
from sklearn import model_selection, metrics, svm, tree, ensemble
import graphviz
from keras.models import Sequential
from keras.layers import Dense


def trees(x_input: pd.DataFrame, y_input: pd.DataFrame, cost_weight, sensitive):
    if not sensitive:
        cost = {0: 1, 1: 1}
    else:
        cost = cost_weight
    model = tree.DecisionTreeClassifier(max_depth=4, class_weight=cost)  # , class_weight=cost_weight
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_input, y_input,
                                                                        test_size=0.1)  # random_state=5
    model = model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    evaluate(y_test, y_predict, cost_weight)
    # viz_tree(model, x_input)


def random_trees(x_input: pd.DataFrame, y_input: pd.DataFrame, cost_weight, sensitive):
    if not sensitive:
        cost = {0: 1, 1: 1}
    else:
        cost = cost_weight
    model = ensemble.RandomForestClassifier(max_depth=3, random_state=0, class_weight=cost)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_input, y_input,
                                                                        test_size=0.1)  # random_state=5
    model = model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    evaluate(y_test, y_predict, cost_weight)
    # viz_tree(model, x_input)


def sv_class(x_input: pd.DataFrame, y_input: pd.DataFrame, cost_weight):
    model = svm.SVC()  # class_weight=cost_weight, max_iter=-1
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_input, y_input,
                                                                        test_size=0.1)  # random_state=5
    model = model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    evaluate(y_test, y_predict, cost_weight)


def keras_nn(x_input: pd.DataFrame, y_input: pd.DataFrame, cost_weight):
    model = Sequential()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_input, y_input,
                                                                        test_size=0.1)  # random_state=5
    model.add(Dense(16, input_dim=38, activation='relu'))  # 39 columns
    model.add(Dense(12, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=64)
    y_predict = model.predict(x_test)
    evaluate(y_test, y_predict, cost_weight)


def viz_tree(model, x):
    # Visualize the trained tree.
    dot_data = sklearn.tree.export_graphviz(model, out_file=None,
                                            filled=True, rounded=True,
                                            special_characters=True,
                                            feature_names=x.columns[:])
    graph = graphviz.Source(dot_data)
    graph.render("DT")


def evaluate(y_test, y_predict, cost_weight):
    print("Accuracy Score: %.4f" % metrics.accuracy_score(y_test, y_predict))
    print("Precision: %.4f" % metrics.precision_score(y_test, y_predict, average="macro"))
    print("Recall: %.4f" % metrics.recall_score(y_test, y_predict, average="macro"))
    print(" F1: %.4f" % metrics.f1_score(y_test, y_predict, average="macro"))
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_predict).ravel()
    print(metrics.confusion_matrix(y_test, y_predict))
    total_cost = cost_weight[1] * fn + cost_weight[0] * fp
    print("Cost: %.f" % total_cost)

    # cv = model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # scores = model_selection.cross_val_score(model, x_input, y_input, scoring='roc_auc', cv=cv, n_jobs=-1)
    # print('Mean ROC AUC: %.3f' % np.mean(scores)
