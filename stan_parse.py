# http://www.paaljapan.org/conference2011/ProcNewest2011/pdf/poster/P-13.pdf

from pycorenlp import StanfordCoreNLP
from pprint import pprint

nlp = StanfordCoreNLP('http://localhost:9000')

text = 'Sarah read the book quickly and understood it correctly'
output = nlp.annotate(text, properties={
    'annotators': 'tokenize,ssplit,pos,depparse,parse',
    'outputFormat': 'json'
})

ADD = 0
for x in output['sentences'][0]['basicDependencies']:
    if x['dep']!='ROOT':
        ADD += abs(x['governor'] - x['dependent'])
ADD /= len(output['sentences'][0]['basicDependencies']) - 1
print(ADD)
