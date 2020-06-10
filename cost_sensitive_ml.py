import pandas as pd
import numpy as np
from imbalance_handling import handle_imbalance
from sklearn import model_selection, metrics, svm, tree, ensemble
# from keras.models import Sequential
# from keras.layers import Dense
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.metrics import savings_score
from sklearn.model_selection import KFold

# MODELS
def ada_trees(x_train, x_test, y_train, y_test, cost_weight, sensitive):
    if not sensitive:
        cost = {0: 1, 1: 1}
    else:
        cost = cost_weight
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, class_weight=cost), n_estimators=5)
    k_fold_evaluation(x_train, y_train, clf,cost_weight)

    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    evaluate(y_test, y_predict, cost_weight)
    return clf

def cost_trees(x_train, x_test, y_train, y_test, cost_weight):

    fp = np.full((y_test.shape[0], 1), cost_weight[0])
    fn = np.full((y_test.shape[0], 1), cost_weight[1])
    tp = np.zeros((y_test.shape[0], 1))
    tn = np.zeros((y_test.shape[0], 1))
    cost_mat_test = np.hstack((fp, fn, tp, tn))

    fp = np.full((y_train.shape[0], 1), cost_weight[0])
    fn = np.full((y_train.shape[0], 1), cost_weight[1])
    tp = np.zeros((y_train.shape[0], 1))
    tn = np.zeros((y_train.shape[0], 1))
    cost_mat_train = np.hstack((fp, fn, tp, tn))

    model = CostSensitiveDecisionTreeClassifier(max_depth=4)
    model = model.fit(x_train.to_numpy(), y_train.to_numpy(), cost_mat_train)
    y_predict = model.predict(x_test.to_numpy())
    evaluate(y_test, y_predict, cost_weight)

def trees(x_train, x_test, y_train, y_test, cost_weight, sensitive):
    if not sensitive:
        cost = {0: 1, 1: 1}
    else:
        cost = cost_weight
    model = tree.DecisionTreeClassifier(max_depth=4, class_weight=cost)  # , class_weight=cost_weight
    k_fold_evaluation(x_train,y_train,model,cost_weight)

    model = model.fit(x_train, y_train)
    # y_predict = model.predict(x_test)
    # evaluate(y_test, y_predict, cost_weight)

    return model

def random_trees(x_train, x_test, y_train, y_test, cost_weight, sensitive):
    if not sensitive:
        cost = {0: 1, 1: 1}
    else:
        cost = cost_weight
    model = ensemble.RandomForestClassifier(max_depth=3, random_state=0, class_weight=cost)
    k_fold_evaluation(x_train, y_train, model,cost_weight)

    model = model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    evaluate(y_test, y_predict, cost_weight)
    return model

def cost_sensitive_re_sampling(x: pd.DataFrame, y: pd.DataFrame, cost_weight):
    count_y = Counter(y)
    major_class = count_y[0]
    minor_class = count_y[1]

    cost_major = int(major_class/cost_weight[1])
    cost_minor = int(major_class/cost_weight[0])
    print("TEST",cost_major, cost_minor)

    sampler = RandomUnderSampler(sampling_strategy={0: cost_major, 1: minor_class}, random_state=1)
    xus, yus = sampler.fit_sample(x, y)

    sampler = RandomOverSampler(sampling_strategy={0: cost_major, 1: cost_minor}, random_state=1)
    xos, yos = sampler.fit_sample(xus, yus)

    print("Cost sensitive re-sample:", Counter(yos))
    return xos,yos

# def sv_class(x_input: pd.DataFrame, y_input: pd.DataFrame, cost_weight):
#     model = svm.SVC()  # class_weight=cost_weight, max_iter=-1
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(x_input, y_input,
#                                                                         test_size=0.1)  # random_state=5
#     model = model.fit(x_train, y_train)
#     y_predict = model.predict(x_test)
#     evaluate(y_test, y_predict, cost_weight)

# def keras_nn(x_input: pd.DataFrame, y_input: pd.DataFrame, cost_weight):
#     model = Sequential()
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(x_input, y_input,
#                                                                         test_size=0.1)  # random_state=5
#     model.add(Dense(16, input_dim=38, activation='relu'))  # 39 columns
#     model.add(Dense(12, activation='relu'))
#     model.add(Dense(2, activation='softmax'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(x_train, y_train, epochs=100, batch_size=64)
#     y_predict = model.predict(x_test)
#     evaluate(y_test, y_predict, cost_weight)

# EVALUATION
def evaluate(y_test, y_predict, cost_weight):
    # print("--- Model Evaluation! --- ")
    # print("Accuracy Score: %.4f" % (metrics.accuracy_score(y_test, y_predict) * 100))
    # print("Precision: %.4f" % (metrics.precision_score(y_test, y_predict, average="macro") * 100))
    # print("Recall: %.4f" % (metrics.recall_score(y_test, y_predict, average="macro") * 100))
    # print("F1: %.4f" % (metrics.f1_score(y_test, y_predict, average="macro") * 100))
    # tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_predict).ravel()
    # print(metrics.confusion_matrix(y_test, y_predict))
    # total_cost = cost_weight[1] * fn + cost_weight[0] * fp
    # print("Cost: %.f" % total_cost)
    # print('\n')
    print("%.4f" % (metrics.accuracy_score(y_test, y_predict) * 100))
    print("%.4f" % (metrics.precision_score(y_test, y_predict, average="macro") * 100))
    print("%.4f" % (metrics.recall_score(y_test, y_predict, average="macro") * 100))
    print("%.4f" % (metrics.f1_score(y_test, y_predict, average="macro") * 100))
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_predict).ravel()
    total_cost = cost_weight[1] * fn + cost_weight[0] * fp
    # print(metrics.confusion_matrix(y_test, y_predict))
    print("%.f" % total_cost)
    print(tp)
    print(fp)
    print(fn)
    print(tn)

def k_fold_evaluation(x,y,model,cost_weight):
    # print("--- KFold Model Evaluation! --- ")
    accuracy_model = []
    precision_model = []
    recall_model = []
    f1_model = []
    total_cost = []
    kf = model_selection.StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    # kf = KFold(n_splits=5, shuffle=False)
    # kf.split(x)
    for train_index, test_index in kf.split(x,y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        accuracy_model.append(metrics.accuracy_score(y_test, y_predict, normalize=True) * 100)
        precision_model.append(metrics.precision_score(y_test, y_predict, average="macro") * 100)
        recall_model.append(metrics.recall_score(y_test, y_predict, average="macro") * 100)
        f1_model.append(metrics.f1_score(y_test, y_predict,average="macro") * 100)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_predict).ravel()
        cost = cost_weight[1] * fn + cost_weight[0] * fp
        total_cost.append(cost)

    # print("K Fold Accuracy scores:", accuracy_model)
    # print("Mean KFold Accuracy: ",round(np.mean(accuracy_model),4))
    print(round(np.mean(accuracy_model), 4))
    # print("K Fold Precision scores:", precision_model)
    # print("Mean K Fold Precision scores:", round(np.mean(precision_model),4))
    print(round(np.mean(precision_model),4))
    # print("K Fold Recall scores:", recall_model)
    # print("Mean K Fold Recall scores:", round(np.mean(recall_model),4))
    print(round(np.mean(recall_model),4))
    # print("K Fold F1 scores:", f1_model)
    # print("Mean KFold F1: ",round(np.mean(f1_model),4))
    print(round(np.mean(f1_model),4))
    print(np.mean(total_cost))

# EXPERIMENTS
def exp_trees(X, Y, cost_weight):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1, random_state=5)

    # print("--- Decision Trees, Re-sampling: NONE, Class Weight: NO ---")
    # trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=False)
    print("--- Decision Trees, Re-sampling: NONE, Class Weight: YES ---")
    ret_tree = trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=True)
    return ret_tree, x_train, x_test, y_test

    print("--- Decision Trees, Re-sampling: OVER, Class Weight: NO ---")
    xO_train, yO_train = handle_imbalance(x_train, y_train, under=False, over=True, combine=False)
    trees(xO_train, x_test, yO_train, y_test, cost_weight, sensitive=False)
    print("--- Decision Trees, Re-sampling: OVER, Class Weight: YES ---")
    trees(xO_train, x_test, yO_train, y_test, cost_weight, sensitive=True)

    print("--- Decision Trees, Re-sampling: Cost-Sensitive, Class Weight: NO ---")
    xcs_train,ycs_train = cost_sensitive_re_sampling(x_train,y_train,cost_weight)
    # xcs_train, xcs_test, ycs_train, ycs_test = model_selection.train_test_split(xcs, ycs, test_size=0.1, random_state=5)
    trees(xcs_train, x_test, ycs_train, y_test, cost_weight, sensitive=False)
    print("--- Decision Trees, Re-sampling: Cost-Sensitive, Class Weight: YES ---")
    trees(xcs_train, x_test, ycs_train, y_test, cost_weight, sensitive=True)

def exp_random_forests(X, Y, cost_weight):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1, random_state=5)

    # print("--- Random Forests, Re-sampling: NONE, Class Weight: NO ---")
    # random_trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=False)
    # print("--- Random Forests, Re-sampling: NONE, Class Weight: YES---")
    # random_trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=True)

    print("--- Random Forests, Re-sampling: OVER, Class Weight: NO ---")
    xO_train, yO_train = handle_imbalance(x_train, y_train, under=False, over=True, combine=False)
    model = random_trees(xO_train, x_test, yO_train, y_test, cost_weight, sensitive=False)
    return model, xO_train, x_test, y_test
    print("--- Random Forests, Re-sampling: OVER, Class Weight: YES ---")
    random_trees(xO_train, x_test, yO_train, y_test, cost_weight, sensitive=True)

    print("--- Random Forests, Re-sampling: Cost-Sensitive, Class Weight: NO ---")
    xcs_train, ycs_train = cost_sensitive_re_sampling(x_train,y_train,cost_weight)
    random_trees(xcs_train, x_test, ycs_train, y_test, cost_weight, sensitive=False)
    print("--- Random Forests, Re-sampling: Cost-Sensitive, Class Weight: YES ---")
    random_trees(xcs_train, x_test, ycs_train, y_test, cost_weight, sensitive=True)

def exp_ada(X,Y,cost_weight):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1, random_state=5)

    # print("--- AdaBoost Decision Trees ---")
    # ada_trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=False)
    print("--- AdaCost Decision Trees ---")
    model = ada_trees(x_train, x_test, y_train, y_test, cost_weight, sensitive=True)
    return model, x_train, x_test, y_test

    print("--- AdaBoost Decision Trees, Re-sampling: OVER ---")
    xO_train, yO_train = handle_imbalance(x_train, y_train, under=False, over=True, combine=False)
    ada_trees(xO_train, x_test, yO_train, y_test, cost_weight, sensitive=False)
    print("--- AdaCost Decision Trees, Re-sampling: OVER ---")
    ada_trees(xO_train, x_test, yO_train, y_test, cost_weight, sensitive=True)

    print("--- AdaBoost Decision Trees, Re-sampling: Cost-Sensitive, Class Weight: NO ---")
    xcs_train, ycs_train = cost_sensitive_re_sampling(x_train,y_train,cost_weight)
    ada_trees(xcs_train, x_test, ycs_train, y_test, cost_weight, sensitive=False)
    print("--- AdaCost Decision Trees, Re-sampling: Cost-Sensitive, Class Weight: YES ---")
    ada_trees(xcs_train, x_test, ycs_train, y_test, cost_weight, sensitive=True)

def exp_cost_tree(X,Y,cost_weight):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1, random_state=5)

    print("--- Cost CLA Decision Trees ---")
    cost_trees(x_train, x_test, y_train, y_test, cost_weight)

    print("--- Cost CLA Decision Trees, Re-sampling: OVER ---")
    xO_train, yO_train = handle_imbalance(x_train, y_train, under=False, over=True, combine=False)
    cost_trees(xO_train, x_test, yO_train, y_test, cost_weight)

    print("--- Cost CLA Decision Trees, Re-sampling: Cost-sensitive ---")
    xcs_train, ycs_train = cost_sensitive_re_sampling(x_train, y_train, cost_weight)
    cost_trees(xcs_train, x_test, ycs_train, y_test, cost_weight)

