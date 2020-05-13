import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

def prep(data):
    # print(data['y'].value_counts())
    # no     36548 , yes     4640

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


def min_max_scale(data):
    minMaxScaler = preprocessing.MinMaxScaler(copy="True", feature_range=(0, 1))
    minMaxScaler.fit(data)
    final = minMaxScaler.transform(data)
    final = pd.DataFrame.from_dict(final)

    pd.set_option('display.expand_frame_repr', False)
    final.columns = data.columns
    print(final)
    return final
