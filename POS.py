"""
This file classifies genres based on n-grams for POS tags
encoding the structural information of a sentence
"""

import numpy as np
import os
import file_reader
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk import pos_tag,pos_tag_sents,BigramTagger
from scipy.spatial import distance
from sklearn import neighbors,datasets
import knn
#
# taglist = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
# pairlist =[]
#
# for i in taglist:
#     for j in taglist:
#         pairlist.append((i,j))
#
filelist,D = file_reader.readfile(30)

# def label(genre):
#     return D[genre]
#
# def featureCreate(bi,lab):
#     global pairlist
#     feature = np.array([lab])
#
#     for i in pairlist:
#
#         if i in bi.keys():
#             feature = np.append(feature,bi[i])
#         else:
#             feature = np.append(feature,0)
#     return feature.reshape(1,-1)
#
# totalcount = {}
# totalF = {}
#
# featureMatrix = np.empty(0)
#
# for genre,files in filelist.items():
#     bi = {}
#     for i in files:
#         f = open(i, 'r')
#
#         tagbook = pos_tag(word_tokenize(f.read()))
#
#         if len(tagbook) > 0:
#             total = 0
#             for i in range(0,len(tagbook)-1) :
#                 x,y = tagbook[i]
#                 x1,y1 = tagbook[i+1]
#
#                 if (y,y1) not in bi.keys():
#                     bi[(y,y1)] = 1
#                     total+=1
#                 else:
#                     bi[(y, y1)] += 1
#                     total+=1
#             for i in bi.keys():
#                 bi[i] = bi[i]/total
#
#         if featureMatrix.size == 0:
#             featureMatrix = featureCreate(bi,label(genre))
#         else:
#             f = featureCreate(bi,label(genre))
#
#             featureMatrix = np.concatenate((featureMatrix,f),axis=0)
#
# np.save('POS_frequency_feat_seq_30',featureMatrix)
#
# print(featureMatrix)
# print(featureMatrix.shape)

#

feat_vec = np.load('POS_frequency_feat_seq_30.npy')

prats = np.load("sen_length_feat_sorted_30.npy")

print(filelist)

add = prats[:,1]

add = np.insert(add,74,[0])
add = np.insert(add,86,[0])

# add = add/np.mean(add)

add = add.reshape(-1,1)

feat_vec = np.concatenate((feat_vec,add),axis=1)
# fiction 1011 - 75,1023 - 87
print(feat_vec.shape)

feat_vec = np.concatenate((feat_vec[:,:1],add/np.mean(add)),axis=1)

accuracy = 0
for i in range(0,100):

    np.random.shuffle(feat_vec)

    sli = 0.8 * len(feat_vec)

    X = np.split(feat_vec, [int(sli)], axis=0)

    y_train = np.transpose(X[0][:, :1])[0]
    X_train = X[0][:, 1:]

    y_test = np.transpose(X[1][:, :1])[0]
    X_test = X[1][:, 1:]

    knn = neighbors.KNeighborsClassifier(n_neighbors=4,weights='distance',metric='manhattan')
    knn.fit(X_train, y_train)

    test = knn.predict(X_test)

    acc = test == y_test
    accuracy += acc.sum() / len(acc)

print(accuracy/100)



