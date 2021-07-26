from readfile import read_corpus

from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def remove_low_frequency_words(threshold, qlist):
    all_words = []
    for q in qlist:
        question = q.split()
        for word in question:
            all_words.append(word)


    counter = Counter(all_words)
    word_freq = counter.most_common()

    word_dic = {}

    for word, freq in word_freq:
        word_dic[word] = freq


    for i in range(len(qlist)):
        wordlist = str(qlist[i]).split()
        for word in wordlist:
            if word_dic[word] < threshold:

                wordlist.remove(word)
        qlist[i] = ' '.join(wordlist)

    return qlist


def remove_punc_num(qlist):
    import string
    import re


    porter_stemmer = PorterStemmer()

    for i in range(len(qlist)):
        l = str.maketrans('', '', string.punctuation)
        newword = str(qlist[i]).translate(l)
        newword = newword.lower()
        newword = porter_stemmer.stem(newword)
        wordlist = newword.split()
        for k in range(len(wordlist)):
            if re.search(r'\d', wordlist[k]):
                wordlist[k] = "#number"
        newword = ' '.join(wordlist)
        qlist[i] = str(newword)

    return qlist


def remove_stopwords(qlist):
    for i in range(len(qlist)):
        wordlist = str(qlist[i]).split()
        filtered = [w for w in wordlist if (w not in stopwords.words('english'))]
        qlist[i] = ' '.join(filtered)
    return qlist

def preprocess(threshold,qlist):
    qlist = remove_punc_num(qlist)
    qlist = remove_stopwords(qlist)
    qlist = remove_low_frequency_words(threshold,qlist)
    return qlist









