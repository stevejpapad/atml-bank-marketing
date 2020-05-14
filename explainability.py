import graphviz
from sklearn.tree import export_text, export_graphviz
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def tree_viz(model, x):
    # Visualize the trained tree.
    dot_data = export_graphviz(model, out_file=None,
                               filled=True, rounded=True,
                               special_characters=True,
                               feature_names=x.columns[:])
    graph = graphviz.Source(dot_data)
    graph.render("DT")


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
    plt.show()

def tree_local_interpretation():
    pass


