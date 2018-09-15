"""
This is a Gaussian Mixture Model Classifier using the Scikit Learn Library
"""

from sklearn import mixture
import numpy as np
import sys


def gmm_classifier(feat):
    feat_vec = np.load(feat)
    np.random.shuffle(feat_vec)

    sli = 0.8 * len(feat_vec)

    X = np.split(feat_vec, [int(sli)], axis=0)

    y_train = np.transpose(X[0][:, :1])[0]
    X_train = X[0][:, 1:]

    y_test = np.transpose(X[1][:, :1])[0]
    X_test = X[1][:, 1:]

    gmm = mixture.GaussianMixture(n_components=5, covariance_type='full')
    gmm.fit(X_train,y_train)
    test = gmm.predict(X_test)

    # knn = neighbors.KNeighborsClassifier(n_neighbors=20)
    # knn.fit(X_train, y_train)
    #
    # test = knn.predict(X_test)

    acc = test == y_test
    accuracy = acc.sum() / len(acc)

    return accuracy


def call_gmm(filename):
    acc_list = np.empty([])
    for x in range(0, 100):
        acc_list = np.append(acc_list, gmm_classifier(filename))
    print("accuracy is ", np.mean(acc_list))


if __name__ == '__main__':
    filename = sys.argv[1]
    call_gmm(filename)
