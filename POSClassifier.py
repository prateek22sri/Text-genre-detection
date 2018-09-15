import numpy as np
import os
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk import pos_tag
from scipy.spatial import distance

taglist = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
pairlist =[]

for i in taglist:
    for j in taglist:
        pairlist.append((i,j))

for g in ['Animals','Children','Fiction','Medicine','Religion']:
    result = []
    bi = {}
    dir = '/home/hduser/ANLP_project/data/'+str(g)

    for root,dirs,files in os.walk(dir,topdown=True):

        # file = "/home/hduser/ANLP_project/data/Animals/"+str(i)
        c = 0
        for name in files:
            fi = os.path.join(root,name)
            # print(c)
            # print(fi)
            if c >= 20:
                break
            c+=1
            f = open(fi, 'r')
            tagbook = pos_tag(word_tokenize(f.read()))

            if len(tagbook) > 0:
                total = 0
                for i in range(0,len(tagbook)-1) :
                    x,y = tagbook[i]
                    x1,y1 = tagbook[i+1]

                    if (y,y1) not in bi.keys():
                        bi[(y,y1)] = 1
                        total+=1
                    else:
                        bi[(y, y1)] += 1
                        total+=1
                for i in bi.keys():
                    bi[i] = bi[i]/total

            final = np.empty(0)


            test = {}

            for i in pairlist:
                test[i] = []
                if i in bi.keys():
                    a = np.array(bi[i]).reshape((1,-1))
                else:
                    a = np.array([0]).reshape((1,-1))

                if final.size == 0:
                    final = a
                else:
                    final = np.concatenate((final, a), axis=0)


            train = np.load('POS_frequency_feat.npy')

            tester=np.transpose(final)
            trained = np.transpose(train)


            d = np.empty(0)
            for i in range(0,5):
                d = np.append(d,distance.cosine(trained[i],tester))

            print(d)
            print(np.argmin(d))
            result.append(np.argmin(d))
    print(g)
    print([result.count(0),result.count(1),result.count(2),result.count(3),result.count(4)])


#animals
# 88
# 2
# 144
# 60
# 0


