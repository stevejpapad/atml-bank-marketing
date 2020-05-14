from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids, NearMiss,TomekLinks
from collections import Counter
from imblearn.combine import SMOTEENN, SMOTETomek

def handle_imbalance(X, Y, under, over, combine):
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
        # nm1 = NearMiss(version=1)
        # X_res, Y_res = nm1.fit_resample(X, Y)
        # print(sorted(Counter(Y_res).items()))
        #

        #Tomek Links
        tl = TomekLinks()
        X_res, Y_res = tl.fit_resample(X, Y)
        print('Resampled dataset shape %s' % Counter(Y_res))
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

    elif combine:
        print('Performing combination')
        print("Before Combination, counts of label '1': {}".format(sum(Y == 1)))
        print("Before Combination, counts of label '0': {} \n".format(sum(Y == 0)))

        #SMOTEENN
        # smote_enn = SMOTEENN(random_state=0)
        # X_res, Y_res = smote_enn.fit_sample(X, Y)
        # print(sorted(Counter(Y_res).items()))
        #

        #SMOTETomek
        smote_tomek = SMOTETomek(random_state=0)
        X_res, Y_res = smote_tomek.fit_resample(X, Y)
        print(sorted(Counter(Y_res).items()))
        #

        print('After Combination, the shape of train_X: {}'.format(X_res.shape))
        print('After Combination, the shape of train_y: {} \n'.format(Y_res.shape))

        print("After Combination, counts of label '1': {}".format(sum(Y_res == 1)))
        print("After Combination, counts of label '0': {}".format(sum(Y_res == 0)))
        return X_res, Y_res


