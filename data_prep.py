import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


def prep(data):
    # print(data['y'].value_counts())
    # no     36548 , yes     4640
    # corr = data.corr()
    # cor_plot = sns.heatmap(corr, annot=True)
    # plt.show()
    # NOTE: y corr: 40% with 'duration', 23% with 'previous' etc

    data[['y']] = data[['y']].replace(['no'], 0)
    data[['y']] = data[['y']].replace(['yes'], 1)
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

    return data


def handle_imbalance(X, Y, under, over, ensemble):
    if under:
        print('Performing undersampling')
    elif over:
        print('Performing oversampling')
        print("Before OverSampling, counts of label '1': {}".format(sum(Y == 1)))
        print("Before OverSampling, counts of label '0': {} \n".format(sum(Y == 0)))
        sm = SMOTE(random_state=2)  # sampling_strategy=1
        X_res, Y_res = sm.fit_sample(X,Y)

        print('After OverSampling, the shape of train_X: {}'.format(X_res.shape))
        print('After OverSampling, the shape of train_y: {} \n'.format(Y_res.shape))

        print("After OverSampling, counts of label '1': {}".format(sum(Y_res == 1)))
        print("After OverSampling, counts of label '0': {}".format(sum(Y_res == 0)))
        return X_res, Y_res
    elif ensemble:
        print('Performing combination')


def min_max_scale(data):
    minMaxScaler = preprocessing.MinMaxScaler(copy="True", feature_range=(0, 1))
    minMaxScaler.fit(data)
    final = minMaxScaler.transform(data)
    final = pd.DataFrame.from_dict(final)
    final.columns = data.columns
    return final
