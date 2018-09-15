"""
This file is the KNN classifier and is used by multiple classifiers after they generate their
respective feature vectors
"""

from sklearn import neighbors
import numpy as np
import sys


def knn_classifier(feat):
    feat_vec = np.load(feat)
    np.random.shuffle(feat_vec)

    sli = 0.8 * len(feat_vec)

    X = np.split(feat_vec, [int(sli)], axis=0)

    y_train = np.transpose(X[0][:, :1])[0]
    X_train = X[0][:, 1:]

    y_test = np.transpose(X[1][:, :1])[0]
    X_test = X[1][:, 1:]

    knn = neighbors.KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train, y_train)

    test = knn.predict(X_test)

    acc = test == y_test
    accuracy = acc.sum() / len(acc)

    return accuracy


def call_knn(filename):
    acc_list = np.empty([])
    for x in range(0, 100):
        acc_list = np.append(acc_list, knn_classifier(filename))
    print("accuracy is ", np.mean(acc_list))


if __name__ == '__main__':
    filename = 'type_token_feat.npy'
    call_knn(filename)
