import graphviz
from sklearn.tree import export_text, export_graphviz
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pdpbox import pdp
from lime.lime_tabular import LimeTabularExplainer

def tree_viz(model, x):
    # Visualize the trained tree.
    dot_data = export_graphviz(model, out_file=None,
                               filled=True, rounded=True,
                               special_characters=True,
                               feature_names=x.columns[:], max_depth=3)
    graph = graphviz.Source(dot_data)
    graph.render("DT")
    return dot_data


def tree_to_text(tree, feature_names):
    r = export_text(tree, feature_names=feature_names)
    print(r)

def tree_feature_importance(model, columns):
    for feature, importance in zip(columns, model.feature_importances_):
        print(f"{feature}: {importance}")

def tree_bar_interpretation(model, x):
    weights = model.feature_importances_
    model_weights = pd.DataFrame({'features': list(x.columns), 'weights': list(weights)})
    model_weights = model_weights.sort_values(by='weights', ascending=False)
    plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    sns.barplot(x="weights", y="features", data=model_weights)
    plt.xticks(rotation=90)
    # plt.show()
    plt.savefig('bar_inter.png', transparent=True)

def tree_local_interpretation(model, x_test, sample, y_pred):
    feature = model.tree_.feature
    threshold = model.tree_.threshold

    node_indicator = model.decision_path(x_test)
    leave_id = model.apply(x_test)
    sample_id = 200  # sample
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
    print()
    print(f'Sample predicred as: {y_pred[sample_id]}')
    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if x_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        print(
            f"decision id node {node_id} : ({list(x_test.columns)[feature[node_id]]} (= {x_test.iloc[sample_id, feature[node_id]]}) {threshold_sign} {threshold[node_id]})")


def tree_dependency_plot(model, dataset):
    pdp_goals = pdp.pdp_isolate(model=model, dataset=dataset, model_features=dataset.columns[0:39],
                                feature='nr.employed')  # 'cons.conf.idx')

    pdp.pdp_plot(pdp_goals, 'Number of employeese')  # 'Consumer confidenc'
    # plt.show()
    plt.savefig('pdp.png', transparent=True)

def lime_local(model, x_train, x_test, y_test):
    explainer = LimeTabularExplainer(
        x_train.values, feature_names=x_train.columns, mode="classification",
        class_names=["non-subscriber", "subscriber"], discretize_continuous=True
    )
    i = 200  # np.random.randint(0, x_test.shape[0])
    exp = explainer.explain_instance(x_test.values[i], model.predict_proba, num_features=len(x_train.columns))
    # exp.as_pyplot_figure()
    print(y_test.values[i])
    names = []
    values = []
    positive = []
    for x in exp.as_list():
        names.append(x[0])
        values.append(x[1])
        if x[1] > 0:
            positive.append('Positive')
        else:
            positive.append('Negative')

    palette = {"Positive": "green", "Negative": "red"}
    plt.figure(figsize=(30, 30), dpi=100)
    sns.barplot(x=values, y=names, hue=positive, palette=palette)
    # plt.show()
    # return exp
    plt.savefig('lime.png', transparent=True)
