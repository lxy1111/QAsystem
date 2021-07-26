
from sklearn.feature_extraction.text import TfidfVectorizer
from readfile import read_corpus
from preprocess import preprocess



def text2vec(qlist):


    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)  # 定义一个tf-idf的vectorizer

    X = vectorizer.fit_transform(qlist)  # 结果存放在X矩阵




    return X, vectorizer

def save_model(model_file,X,vectorizer,qlist):
    import pickle
    from collections import defaultdict

    inverted_index = defaultdict(list)

    for i in range(len(qlist)):
        q = qlist[i]
        wordset = set(q.split())
        for word in wordset:
            inverted_index[word].append(i)

    with open(model_file, 'wb') as f:
        pickle.dump(X, f)
        pickle.dump(vectorizer,f)
        pickle.dump(qlist,f)
        pickle.dump(inverted_index,f)



# model_file = './data/QAModel.pkl'
#
# originalqlist,originalalist = read_corpus()
#
# qlist = preprocess(10,originalqlist)
#
# X,vectorizer = text2vec(qlist)
#
# save_model(model_file,X,vectorizer,qlist)

