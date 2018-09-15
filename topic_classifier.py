import numpy as np

feat_vec =np.load("Topic_feat.npy")

accuracy = 0
for i in range(0,50):

    np.random.shuffle(feat_vec)

    sli = 0.8*len(feat_vec)

    X = np.split(feat_vec,[int(sli)],axis=0)

    y_train = np.transpose(X[0][:,:1])[0]
    X_train = X[0][:,1:]

    y_test = np.transpose(X[1][:,:1])[0]
    X_test = X[1][:,1:]

    lab_test = []

    for i in X_test:
        max = []
        for j in X_train:
            r = i&j
            max.append(np.sum(r))

        max = np.array(max)
        lab_test.append(y_train[np.argmax(max)])


    acc = np.array(lab_test)==y_test
    accuracy += acc.sum()/len(acc)

print(accuracy/50)




#
# knn = neighbors.KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
#
# test = knn.predict(X_test)
#
# acc = test == y_test
# accuracy = acc.sum()/len(acc)
#
# print(accuracy)
#
#
#
# print(feat.shape)


