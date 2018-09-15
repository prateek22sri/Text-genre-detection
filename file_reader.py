"""
This file is a standard file reading script to maintain uniformity for all classifiers
"""

import os
from pprint import pprint


def readfile(num=0,random=False):
    categories = {}
    file_structure = {}
    for root, dirs, files in os.walk("./data", topdown=True):
        dirs.sort()
        for name in dirs:
            file_structure[name] = []
            d = os.path.join(root, name)
            for roots, dir1, files1 in os.walk(d, topdown=True):
                if not random:
                    files1.sort()
                i = 0
                for names in files1:
                    if num > 0:
                        if i < num:
                            file_structure[name].append(os.path.join(roots, names))
                            if num != 0:
                                i += 1
                    else:
                        file_structure[name].append(os.path.join(roots, names))

    ctr = 0
    for i in file_structure.keys():
        categories[i] = ctr
        ctr += 1

    return file_structure, categories


if __name__ == '__main__':
    pprint(readfile(num=5,random=True))
