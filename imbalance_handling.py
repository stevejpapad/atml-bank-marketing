from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NearMiss
from collections import Counter

def handle_imbalance(X, Y, under, over, ensemble):
    if under:
        print('Performing undersampling')
        print("Before Undersampling, counts of label '1': {}".format(sum(Y == 1)))
        print("Before Undersampling, counts of label '0': {} \n".format(sum(Y == 0)))

        # Cluster Centroids

        # cc = ClusterCentroids(random_state=0)
        # X_res, Y_res = cc.fit_sample(X, Y)
        # print(sorted(Counter(Y_res).items()))
        #
        # NearMiss
        nm1 = NearMiss(version=1)
        X_res, Y_res = nm1.fit_resample(X, Y)
        print(sorted(Counter(Y_res).items()))
        #

        print('After Undersampling, the shape of train_X: {}'.format(X_res.shape))
        print('After Undersampling, the shape of train_y: {} \n'.format(Y_res.shape))

        print("After Undersampling, counts of label '1': {}".format(sum(Y_res == 1)))
        print("After Undersampling, counts of label '0': {}".format(sum(Y_res == 0)))
        return X_res, Y_res

    elif over:
        print('Performing oversampling')
        print("Before OverSampling, counts of label '1': {}".format(sum(Y == 1)))
        print("Before OverSampling, counts of label '0': {} \n".format(sum(Y == 0)))

        # SMOTE
        sm = SMOTE(random_state=2)  # sampling_strategy=1
        X_res, Y_res = sm.fit_sample(X, Y)
        #

        print('After OverSampling, the shape of train_X: {}'.format(X_res.shape))
        print('After OverSampling, the shape of train_y: {} \n'.format(Y_res.shape))

        print("After OverSampling, counts of label '1': {}".format(sum(Y_res == 1)))
        print("After OverSampling, counts of label '0': {}".format(sum(Y_res == 0)))
        return X_res, Y_res

    elif ensemble:
        print('Performing combination')