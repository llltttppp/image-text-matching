import numpy as np
import pandas as pd
import cPickle as pkl
#vect_file = '/home/ltp/WorkShop/fastText/model/ourword2vec.vec'
#out_pklfile = './model/word2vec/ourword2vec.pkl'
#vectors = [np.array([float(g) for g in v.strip().split(' ')[1:]]) for v in open(vect_file,'r').readlines() ][1:]
#words = [v.strip().split(' ')[0] for v in open(vect_file,'r').readlines()][1:]
#word2vect={}
#word2vect.update(zip(words,vectors))
#for v in word2vect.keys():
    #word2vect[v]=word2vect[v]/np.linalg.norm(word2vect[v])
#pkl.dump(word2vect,open(out_pklfile,'w'))
word2vect =pkl.load(open('./model/word2vec/ourword2vec.pkl'))
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
    
    
most_similar('聚智堂')