# -*- coding:utf-8 -*- 
import numpy as np
import pandas as pd
import thulac
import tensorflow as tf
import h5py
from sklearn.decomposition import PCA,FastICA
from sklearn.externals import joblib 
import gensim
import sys
sys.path.append('./yael')
import yael
import time
import cPickle as pkl
root = '/media/ltp/40BC89ECBC89DD32/souhu_fusai/2016News/其他/'
seq_root = '/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_seg/'

#match_list =[v.split(' '*6) for v in  open('trainMatching_shuffle_filter.txt','r').readlines()]
word2vec_model=pkl.load(open('./model/word2vec/word2vec_11w.pkl'))
#word_sample =np.random.randint(0,len(word2vec_model.syn0norm),12000)
#word_result =ica.fit_transform(word2vec_model.syn0norm[word_sample])
def generate_nvocabulary(raw_vocabulary='./vocabulary/vocabulary_nv.txt',outfile= './vocabulary/vocabulary_n.txt'):
    seg_model = thulac.thulac(seg_only=False)    
    word_list = open(raw_vocabulary,'r').readlines()
    with open(outfile,'w') as fout:
        for v in word_list:
            word = v.strip().split(' ')[0]
            if 'n' in seg_model.cut(word)[0][1]:
                fout.writelines([v])
            
def generate_nostopwords_vocabulary(raw_vocabulary = './vocabulary/vocabulary_nv_4w.txt',outfile=None,stopwords='./vocabulary/stopwords.txt'):
    if not outfile:
        outfile = raw_vocabulary[:-4]+'_nostopwords.txt'
    stopwords_list = [v.strip() for v in open(stopwords,'r').readlines()]
    with open(outfile,'w') as fout:
        for v in open(raw_vocabulary,'r').readlines():
            word = v.strip().split(' ')[0]
            if word not in stopwords_list:
                fout.writelines([v])
def generate_icamodel(train_vocabulary='./vocabulary/vocabulary_nv_4w_nostopwords.txt',model_path='./model/ICA/ica.model'):
    train_vocab =[v.strip() for v in open(train_vocabulary,'r').readlines()]
    train_sample = np.zeros([len(train_vocab),300])
    for i,v in enumerate(train_vocab):
        word = v.split(' ')[0]
        try:
            train_sample[i]= word2vec_model[word]
        except:
            print word
    ica = FastICA(300,max_iter=800)
    ica.fit(train_sample)
    joblib.dump(ica,model_path)
    
    pass

def generate_rawvocabulary(vocabulary='./vocabulary/vocabulary.txt',outfile='./vocabulary/vocabulary_nofreq_11w.txt',n=115000):
    vocab =[v.strip() for v in open(vocabulary,'r').readlines()]
    with open(outfile,'w') as fout:
        for v in vocab[:n]:
            fout.writelines([v.split(' ')[0]+'\n'])
def generate_gmmmodel(centersnum=30,icamodel='./model/ICA/ica.model',model_path = './model/GMM/gmm.model',train_vocabulary='./vocabulary/vocabulary_nv_4w_nostopwords.txt'):
    ica = joblib.load(icamodel)
    train_vocab =[v.strip() for v in open(train_vocabulary,'r').readlines()]
    train_sample = np.zeros([len(train_vocab),300])
    for i,v in enumerate(train_vocab):
        word = v.split(' ')[0]
        try:
            train_sample[i]= word2vec_model[word]
        except:
            print word    
    gmm_sample = ica.transform(train_sample).astype(np.float32) 
    gmm_model = yael.ynumpy.gmm_learn(gmm_sample,centersnum,niter=50,redo=2)
    pkl.dump(gmm_model,open(model_path,'w'))
def generate_fishervector(sample_set,ica_model='./model/ICA/ica.model',gmm_model_path='./model/GMM/gmm.model'):
    ica = joblib.load(ica_model)
    gmm_model =pkl.load(open(gmm_model_path,'r'))
    centenrs = gmm_model[0].shape[0]
    dims = gmm_model[1].shape[1]
    fishervector = np.zeros([len(sample_set),centenrs*dims*2])
    for i,v in enumerate(sample_set):
        words =v.strip().split(' ')
        vectors =[]
        for j in words:
            try:
                vectors.append(word2vec_model[j])
            except:
                print 'Not found %s'%j
        fishervector[i]=yael.ynumpy.fisher(gmm_model,ica.transform(np.array(vectors)).astype(np.float32) ,include='mu sigma')
    return fishervector
    
if __name__ == '__main__':
    sentence = np.array([v.strip() for v in open('./train_txt/train.txt').readlines()])
    hout =h5py.File('/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_sentence_fishervectors.h5','w')
    outdata=hout.create_dataset('feature',shape=[len(sentence),18000],dtype=np.float32)
    L=len(sentence)
    batch_size=10000
    N=len(sentence)/batch_size
    for v in range(N+1):
        if (v+1)*batch_size>L:
            sample_set=sentence[v*batch_size:]
            outdata[v*batch_size:,:]=generate_fishervector(sample_set)
        else:
            sample_set=sentence[v*batch_size:(v+1)*batch_size]
            outdata[v*batch_size:(v+1)*batch_size,:]=generate_fishervector(sample_set)
        print 'No.%d batch is done'%v
    hout.close()
    pass