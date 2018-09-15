"""
This file creates feature vectors using frequency distribution of hypernyms and lemmas to train
"""

import numpy as np
from nltk import pos_tag, word_tokenize
import file_reader
from pprint import pprint
from collections import Counter
from nltk.corpus import wordnet
from collections import OrderedDict
from copy import deepcopy
from math import ceil
from random import shuffle


# function to return test and training sets
# def make_train_test_set(n_files=5, random=True):
#     file_structure_train = OrderedDict()
#     file_structure_test = OrderedDict()
#     file_structure, categories = file_reader.readfile(num=n_files)
#     for genre in file_structure.keys():
#         if random:
#             shuffle(file_structure[genre])
#         file_structure_test[genre] = file_structure[genre][:ceil(len(file_structure[genre]) / 8)]
#         file_structure_train[genre] = file_structure[genre][ceil(len(file_structure[genre]) / 8):]
#     return file_structure_train, file_structure_test, categories


def train(file_structure, cat):
    unreadable_files = []
    val_vec = []
    for folder in file_structure.keys():
        word_vec = []
        for file in file_structure[folder]:
            new_word_list = []
            try:
                text = open(file).read().replace(u'\ufeff', '').replace(u'\n', ' ')
            except:
                unreadable_files.append(file)
                continue

            if len(text) == 0:
                unreadable_files.append(file)
                continue

            tokenized_text = word_tokenize(text)
            num_tokens = len(tokenized_text)
            tmp = pos_tag(tokenized_text)
            filtered_tok_text = list(filter(lambda x: ('NN' in x[1]) and len(x[0]) > 1, tmp))
            for x in filtered_tok_text:
                if len(x[0]) == 1:
                    print(x)

            (tok, tag) = zip(*filtered_tok_text)
            tok_count = Counter(tok)
            tok_list, count = zip(*tok_count.most_common(10))
            # print(tok_list)
            count = np.array(count)/num_tokens


            # create a new word list with the words, hypernyms and lemmas

            for i in range(0,len(tok_list)):
                new_word_list.append(tok_list[i])
                try:
                    new_word_list.append(wordnet.synsets(tok_list[i])[0].lemma_names('eng')[0])
                    count = np.append(count, count[i])
                except:
                    pass

                try:
                    new_word_list.append(wordnet.synsets(tok_list[i])[0].hypernyms()[0].lemma_names('eng')[0])
                    count = np.append(count, count[i])
                except:
                    pass

            print(len(new_word_list))
            print(count.shape)
            exit(1)

    return None, None, None


def test(file_structure, lv):
    unreadable_files = []
    token_count = {x: 0 for x in lv}
    vec_dict = OrderedDict()
    for folder in file_structure.keys():
        for file in file_structure[folder]:
            train_vector = []

            try:
                text = open(file).read().replace(u'\ufeff', '').replace(u'\n', ' ')
            except:
                unreadable_files.append(file)
                continue

            if len(text) == 0:
                unreadable_files.append(file)
                continue

            # text = text.translate({ord(c): " " for c in "!@#$%^&*()[]{};:,/<>?\|`~-=_+"})
            tokenized_text = word_tokenize(text)
            tmp = pos_tag(tokenized_text)
            filtered_tok_pos_text = list(filter(lambda x: ('NN' in x[1]), tmp))
            filtered_tok_text, filtered_pos = zip(*filtered_tok_pos_text)
            for word in filtered_tok_text:
                train_vector.append(word)
                try:
                    train_vector.append(wordnet.synsets(word)[0].lemma_names('eng')[0])
                except:
                    pass

                try:
                    train_vector.append(wordnet.synsets(word)[0].hypernyms()[0].lemma_names('eng')[0])
                except:
                    pass

            for word in lv:
                try:
                    token_count[word] = train_vector.count(word)
                except:
                    token_count[word] = 0

            vec_dict[file] = token_count
    return vec_dict, unreadable_files


if __name__ == '__main__':
    # file_structure_train, file_structure_test, categories = make_train_test_set(n_files=5, random=True)
    val_dict_train, unreadable_files_train, label_vector = train(file_structure_train, categories)
    # vec_dict, unreadable_files_test = test(file_structure_test, label_vector)
