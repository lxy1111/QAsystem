from nltk.corpus import stopwords
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from readfile import read_corpus
from preprocess import preprocess
from tfidf import text2vec


def top5results(input_q, vectorizer, originalqlist, originalalist, X):
    from nltk.stem.porter import PorterStemmer
    import numpy as np
    from queue import PriorityQueue

    l = str.maketrans('', '', string.punctuation)
    inputstr = input_q.translate(l)
    inputstr = inputstr.lower()

    inputlist = inputstr.split()
    filtered = [w for w in inputlist if (w not in stopwords.words('english'))]
    inputstr = ' '.join(filtered)

    porter_stemmer = PorterStemmer()
    s = porter_stemmer.stem(inputstr)
    slist = s.split()

    for i in range(len(slist)):
        if re.search(r'\d', slist[i]):
            slist[i] = "#number"
    inputstr = ' '.join(slist)

    inputarray = vectorizer.transform([inputstr])  # 结果存放在X矩阵

    inputarray = inputarray.toarray()

    res_dic = {}
    i = 0
    for v in X.toarray():
        res_dic[i] = np.dot(np.array(v), np.array(inputarray[0])) / (
                np.linalg.norm(np.array(v)) * np.linalg.norm(np.array(inputarray[0])))
        i += 1

    # print(res_dic[-5:])
    res = sorted(res_dic.items(), key=lambda x: x[1], reverse=True)

    top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表
    # hint: 利用priority queue来找出top results. 思考为什么可以这么做？
    result = []
    cnt = 0
    for i, s in res[0:5]:
        result.append(originalalist[i])
        print(cnt)
        print('question' + ": " + originalqlist[i])
        print('similarity: ' + str(s))
        print('answer: ' + originalalist[i])
        print()
        cnt += 1

    return result  # 返回相似度最高的问题对应的答案，作为TOP5答案


def top5results_invidx(input_q, vectorizer, originalqlist, originalalist, X, inverted_index):
    from nltk.stem.porter import PorterStemmer
    import numpy as np
    from queue import PriorityQueue

    l = str.maketrans('', '', string.punctuation)
    inputstr = input_q.translate(l)
    inputstr = inputstr.lower()

    inputlist = inputstr.split()
    filtered = [w for w in inputlist if (w not in stopwords.words('english'))]
    inputstr = ' '.join(filtered)

    porter_stemmer = PorterStemmer()
    s = porter_stemmer.stem(inputstr)
    slist = s.split()

    for i in range(len(slist)):
        if re.search(r'\d', slist[i]):
            slist[i] = "#number"
    inputstr = ' '.join(slist)

    inputarray = vectorizer.transform([inputstr])  # 结果存放在X矩阵

    inputarray = inputarray.toarray()

    possibledoc = []
    for word in set(slist):
        possibledoc += inverted_index[word]
    possibledoc = list(set(possibledoc))

    res_dic = {}

    allsentencearray = X.toarray()

    for index in possibledoc:
        v = allsentencearray[index]
        res_dic[index] = np.dot(np.array(v), np.array(inputarray[0])) / (
                np.linalg.norm(np.array(v)) * np.linalg.norm(np.array(inputarray[0])))

    # print(res_dic[-5:])
    res = sorted(res_dic.items(), key=lambda x: x[1], reverse=True)

    top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表
    # hint: 利用priority queue来找出top results. 思考为什么可以这么做？
    result = []
    cnt = 0
    for i, s in res[0:5]:
        result.append(originalalist[i])
        print(cnt)
        print('question' + ": " + originalqlist[i])
        print('similarity: ' + str(s))
        print('answer: ' + originalalist[i])
        print()
        cnt += 1



def top5results_emb(input_q, originalqlist, originalalist,  inverted_index,words_dictionary, res):

    from nltk.stem.porter import PorterStemmer
    import numpy as np
    from queue import PriorityQueue

    inputlist = input_q.split()
    filtered = [w for w in inputlist if (w not in stopwords.words('english'))]
    inputstr = ' '.join(filtered)

    l = str.maketrans('', '', string.punctuation)
    inputstr = inputstr.translate(l)
    inputstr = inputstr.lower()

    porter_stemmer = PorterStemmer()
    s = porter_stemmer.stem(inputstr)
    slist = s.split()

    for i in range(len(slist)):
        if slist[i].isdigit():
            slist[i] = "#number"
    inputstr = ' '.join(slist)

    vec = []
    for word in slist:
        if word in words_dictionary:
            vec.append(words_dictionary[word])

    tmp = [0] * 100
    for v in vec:
        for j in range(len(v)):
            tmp[j] += float(v[j])
    if len(vec) > 0:
        tmp = [x / len(vec) for x in tmp]

    inputarray = tmp

    possibledoc = []
    for word in set(slist):
        possibledoc += inverted_index[word]
    possibledoc = list(set(possibledoc))

    res_dic = {}

    for index in possibledoc:
        v = res[index]
        res_dic[index] = np.dot(np.array(v), np.array(inputarray)) / (
                    np.linalg.norm(np.array(v)) * np.linalg.norm(np.array(inputarray)))

    # print(res_dic[-5:])
    sortedres = sorted(res_dic.items(), key=lambda x: x[1], reverse=True)


    result = []
    cnt = 0
    for i, s in sortedres[0:5]:
        result.append(originalalist[i])
        print(cnt)
        print('question' + ": " + originalqlist[i])
        print('similarity: ' + str(s))
        print('answer: ' + originalalist[i])
        print()
        cnt += 1






originalqlist, originalalist = read_corpus()

import pickle

model_file = './data/QAModel.pkl'


with open(model_file, 'rb') as f:
    X = pickle.load(f)
    vectorizer = pickle.load(f)
    qlist = pickle.load(f)
    inverted_index = pickle.load(f)




import sys

text = sys.argv[1]
mode = sys.argv[2]


if mode == 'tfidf':
    top5results_invidx(text, vectorizer, originalqlist, originalalist, X, inverted_index)
elif mode == 'emb':
    model_file2 = './data/glove.pkl'
    with open(model_file2, 'rb') as f:
        words_dictionary = pickle.load(f)
        res = pickle.load(f)
    top5results_emb(text, originalqlist, originalalist, inverted_index,words_dictionary,res)






