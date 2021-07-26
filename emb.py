
from readfile import read_corpus
from preprocess import preprocess

f2 = open("data/glove.6B/glove.6B.100d.txt","r")
lines = f2.readlines()
results = []
for line in lines:
    wordlist = line.split()
    results+=wordlist

words_dictionary = {}
for i in range(0,len(results),101):
    word = results[i]
    vec = results[i+1:i+101]
    words_dictionary[word] = vec

import numpy as np

originalqlist,originalalist = read_corpus()

qlist = preprocess(10,originalqlist)



newX = np.zeros((len(qlist), 100))
res = []
for i in range(len(qlist)):
    q = qlist[i]
    wordlist = q.split()
    vec = []
    for word in wordlist:
        if word in words_dictionary:
            vec.append(words_dictionary[word])
    tmp = [0] * 100
    for v in vec:
        for j in range(len(v)):
            tmp[j] += float(v[j])
    if len(vec) > 0:
        tmp = [x / len(vec) for x in tmp]
    res.append(tmp)

import pickle

model_file = './data/glove.pkl'

with open(model_file, 'wb') as f:
    pickle.dump(words_dictionary,f)
    pickle.dump(res,f)


