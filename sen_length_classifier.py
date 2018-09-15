# on the basis of sentence length

import numpy as np
import nltk
import file_reader
from pprint import pprint

X = np.array([])

categories = {}
flag = False

unreadable_files = []

mean_sen_len_cat = []

file_structure, categories = file_reader.readfile(num=30,random=False)

for folder in file_structure.keys():
    avg_sen_len_vec = []
    for file in file_structure[folder]:
        try:
            text = open(file).read().lower().replace(u'\ufeff', '').replace(u'\n', ' ')

        except:
            unreadable_files.append(file)
            # print("Couldn't read ",file)
            continue

        if len(text) == 0:
            unreadable_files.append(file)
            # print(file, "skipped")
            continue

        sentences = nltk.sent_tokenize(text)

        sen_len_vec = np.array([])
        for sentence in sentences:
            length = len(nltk.word_tokenize(sentence))
            if length != 0:
                sen_len_vec = np.append(sen_len_vec, length)
            else:
                continue

        mean_sentence_length = np.mean(sen_len_vec)
        X = np.append(X, categories[folder])
        X = np.append(X, mean_sentence_length)
        avg_sen_len_vec = np.append(avg_sen_len_vec, mean_sentence_length)

    print(folder, " : ", np.mean(avg_sen_len_vec))
    mean_sen_len_cat.append(np.mean(avg_sen_len_vec))

# print(mean_sen_len_cat)
X = X.reshape(-1, 2)
# print(X)
np.save('sen_length_feat_sorted_30', X)

print("Could not read the following files:")
print("==================================")
pprint(unreadable_files)

"""
Mean sentence length for different genres
=========================================
Religion  :  28.9276182665
Animals  :  24.4105895584
Medicine  :  27.850378578
Fiction  :  26.4388886411
Children  :  23.0763251987
"""
