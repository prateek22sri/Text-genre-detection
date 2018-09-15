"""
This file calculates accuracy of type token classifier from the complexity_vector_creation.py
"""

# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.datasets import load_files
# import knn
#
# vocab = np.load('vocab.npy')
# len_vec = np.load('len_vec_sorted.npy')
# dataset = load_files('data/')
# count_vect = CountVectorizer(vocabulary=list(vocab[0]))
# X = count_vect.fit_transform(dataset.data)
# # print(X.shape, len_vec.shape)
# # np.save('type_token_feat', X)
# len_vec = len_vec + 0.000000000001
# tmp = X / len_vec
# np.save('type_token_feat', tmp)
# # print(np.divide(X,len_vec[:, None]))
# # X = np.load('type_token_feat.npy')
# # print(tmp.toarray())

import numpy as np
from sklearn import neighbors

feat_vec = np.load("type_token_feat.npy")
np.random.shuffle(feat_vec)

sli = 0.8 * len(feat_vec)

X = np.split(feat_vec, [int(sli)], axis=0)
# print(X.shape)

y_train = np.transpose(X[0][:, :1])[0]
X_train = X[0][:, 1:]
print(X_train.shape, y_train.shape)
y_test = np.transpose(X[1][:, :1])[0]
X_test = X[1][:, 1:]
print(X_test.shape, y_test.shape)
knn = neighbors.KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)

test = knn.predict(X_test)
acc = test == y_test
accuracy = acc.sum() / len(acc)

print(accuracy)
