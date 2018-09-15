"""
This file uses the complexity_vector_creation.py sort the vectors in scikit learn format and save the numpy arrays
"""

import numpy as np

vocab = np.load('vocab.npy')
file_list = np.load('file_list.npy')
len_vec = np.load('len_vec.npy')

a = list(file_list[0])

tmp = {
    'Animals':0,
    'Children':10000,
    'Fiction':20000000000,
    'Medicine':3000000000000000,
    'Religion':4000000000000000000000000
}

new_a = []
for i in a:
    new_a.append(int(str(tmp[i.split('/')[2]])+str(i.split('/')[3])))

X = list(len_vec.transpose()[0])

abc = np.array([x for _,x in sorted(zip(new_a,X))]).reshape(1,-1).transpose()
np.save('len_vec_sorted',abc)