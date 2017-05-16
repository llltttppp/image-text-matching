import numpy as np
import pandas as pd
import cPickle as pkl
vectors = [np.array([float(g) for g in v.strip().split(' ')[1:]]) for v in open('./vocabulary/word2vec_11w.txt','r').readlines() ]
words = [v.strip().split(' ')[0] for v in open('./vocabulary/word2vec_11w.txt','r').readlines() ]
word2vect={}
word2vect.update(zip(words,vectors))
for v in word2vect.keys():
    word2vect[v]=word2vect[v]/np.linalg.norm(word2vect[v])
#pkl.dump(word2vect,open('word2vec_11w.pkl','w'))

def most_similar(word,n=10):
    word_list = word2vect.keys()
    sim =np.zeros((len(word_list),))
    vector = word2vect[word]
    for i,v in enumerate(word_list):
        sim[i] = word2vect[v].dot(vector)/np.linalg.norm(vector)/np.linalg.norm(word2vect[v])
    rank = np.argsort(-sim)
    for i in rank[:n]:
        print word_list[i]
    print rank[:n]
    
    
most_similar('神雕侠侣')