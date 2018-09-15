"""
This file creates vocabulary for the dataset,
length vector and file list so that it could be used to calculate complexity
"""

import file_reader
from nltk import word_tokenize, pos_tag, wordnet
import numpy as np

fs, cat = file_reader.readfile()

# word_list = set()
len_list = []
filename_list = []
for folder in fs.keys():
    for file in fs[folder]:
        text = open(file).read().replace(u'\ufeff', '').replace(u'\n', ' ')
        if len(text) > 0:
            tokenized_text = word_tokenize(text)
            len_list.append(len(tokenized_text))
            filename_list.append(file)
        else:
            len_list.append(0)
            filename_list.append(file)
            # tmp = pos_tag(tokenized_text)

            # filtered_tok_text = list(filter(lambda x: ('NN' in x[1]) and len(x[0]) > 1, tmp))
            # if len(filtered_tok_text) > 0:
            #     (tok, tag) = zip(*filtered_tok_text)
            #     tok_count = Counter(tok)
            #     tok_list, count = zip(*tok_count.most_common(10))
            #     word_list.update(tok_list)

# word_list = np.array(list(word_list)).reshape(1, -1)
len_list = np.array(len_list).reshape(1, -1).transpose()
file_list = np.array(filename_list).reshape(1, -1)
np.save('len_vec', len_list)
np.save('file_list', file_list)
# np.save('vocab', word_list)

