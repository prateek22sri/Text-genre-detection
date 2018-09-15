"""
This file classifies genre based on TF-IDF

With lemmatization of topics : 87%
Without lemmatization of topics: 91%

Varying number of topics per document in train set:
    Top 100 topics: 87%
    Top 50 topics: 89%
    Top 5 topics: 84%
"""


from __future__ import division
import file_reader as file
import math
import numpy as np
from nltk.corpus import stopwords as stop
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def tfidf(text,label):
    StopWords=list(stop.words('english'))
    metadata=["gutenberg","project","said"]
    stops = ['about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already',
            'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything',
            'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b',
            'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'been', 'before',
            'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came',
            'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did',
            'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'down', 'downed', 'downing', 'downs',
            'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly',
            'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts',
            'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further',
            'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given',
            'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped',
            'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'herself', 'high', 'high',
            'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in',
            'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k',
            'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later',
            'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made',
            'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly',
            'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs',
            'never', 'new', 'new', 'newer', 'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now',
            'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one',
            'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other',
            'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place',
            'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting',
            'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'right',
            'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed',
            'seeming', 'seems', 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows',
            'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something',
            'somewhere', 'state', 'states', 'still', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that',
            'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks',
            'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together',
            'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon',
            'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways',
            'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole',
            'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'x',
            'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours']

    StopWords=StopWords+metadata+stops

    #tfidf
    sklearn_tfidf = TfidfVectorizer(input='filename',stop_words=StopWords)
    s=sklearn_tfidf.fit_transform(text.keys())
    s=s.toarray()
    t=sklearn_tfidf.get_feature_names()

    x=[]
    for i in s:
        temp=''
        for j in i.argsort()[-100:][::-1]:
            temp+=" "+lemmatizer.lemmatize(t[j])

        if len(x)==0:
            x=[temp]
        else:
            x.append(temp)

    cts = CountVectorizer(input='content',binary=True)

    A = cts.fit_transform(x)
    r = A.toarray()

    r = np.insert(r,0,label,axis=1)
    np.save("Topic_feat",r)


if __name__ == '__main__':
    file_structure,cat=file.readfile(300)
    label = []
    unreadable_files = []
    text={}
    for folder in file_structure.keys():
        for file in file_structure[folder]:

            try:
                text[file] = open(file).read().lower().replace(u'\ufeff', '').replace(u'\n', ' ')
                label.append(cat[folder])
            except:
                unreadable_files.append(file)
                print("Couldn't read ", file)
                continue

            if len(text) == 0:
                unreadable_files.append(file)
                print(file, "skipped")
                continue
    tfidf(text,label)