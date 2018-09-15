"""
This file is a complexity classifier based off of the library textstat
The accuracy of this classifier is around 30% (after averaging 100 runs).
It variation extends from 25 to 35%.

"""

# http://www.paaljapan.org/conference2011/ProcNewest2011/pdf/poster/P-13.pdf
# https://raventools.com/blog/ultimate-list-of-online-content-readability-tests/

import numpy as np
import file_reader
from pprint import pprint
from textstat.textstat import textstat


X = np.array([])

categories = {}
flag = False

unreadable_files = []

mean_sen_len_cat = []

file_structure, categories = file_reader.readfile(num=30)

for folder in file_structure.keys():
    avg_sen_len_vec = []
    for file in file_structure[folder]:
        try:
            text = open(file).read().replace(u'\ufeff', '').replace(u'\n', ' ')
        except:
            unreadable_files.append(file)
            continue

        if len(text) == 0:
            unreadable_files.append(file)
            continue

        complexity = textstat.dale_chall_readability_score(text)
        X = np.append(X, categories[folder])
        X = np.append(X, complexity)
        avg_sen_len_vec = np.append(avg_sen_len_vec, complexity)

    print(folder, " : ", np.mean(avg_sen_len_vec))
    mean_sen_len_cat.append(np.mean(avg_sen_len_vec))

X = X.reshape(-1, 2)
np.save('sen_comp_feat_sorted_30', X)

print("Could not read the following files:")
print("==================================")
pprint(unreadable_files)
