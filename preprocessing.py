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
import Queue
import multiprocessing
root = '/media/ltp/40BC89ECBC89DD32/souhu_fusai/2016News/其他/'
seq_root = '/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_seg/'

#match_list =[v.split(' '*6) for v in  open('trainMatching_shuffle_filter.txt','r').readlines()]
word2vec_model=pkl.load(open('./model/word2vec/ourword2vec.pkl'))
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
def generate_icamodel(train_vocabulary='./vocabulary/vocabulary_nv_4w.txt',model_path='./model/ICA/ica_ourword2vec.model'):
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
def generate_gmmmodel(centersnum=30,icamodel='./model/ICA/ica_ourword2vec.model',model_path = './model/GMM/gmm_ourword2vec.model',train_vocabulary='./vocabulary/vocabulary_nv_4w.txt'):
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
    gmm_model = yael.ynumpy.gmm_learn(gmm_sample,centersnum,niter=50,redo=1)
    pkl.dump(gmm_model,open(model_path,'w'))
def generate_rawgmmmodel(centersnum=1000,model_path = './model/GMM/gmm_1000.model',train_vocabulary='./vocabulary/vocabulary_nv_4w.txt'):
    train_vocab =[v.strip() for v in open(train_vocabulary,'r').readlines()]
    train_sample = np.zeros([len(train_vocab),300])
    for i,v in enumerate(train_vocab):
        word = v.split(' ')[0]
        try:
            train_sample[i]= word2vec_model[word]
        except:
            print word    
    gmm_sample =train_sample.astype(np.float32) 
    gmm_model = yael.ynumpy.gmm_learn(gmm_sample,centersnum,niter=30,redo=5)
    pkl.dump(gmm_model,open(model_path,'w'))
def generate_fishervector(sample_set,ica_model='./model/ICA/ica_ourword2vec.model',gmm_model_path='./model/GMM/gmm_ourword2vec.model',max_num = 30000):
    ica = joblib.load(ica_model)
    gmm_model =pkl.load(open(gmm_model_path,'r'))
    centenrs = gmm_model[0].shape[0]
    dims = gmm_model[1].shape[1]
    fishervector = np.zeros([len(sample_set),centenrs*dims*2])+0.00001
    for i,v in enumerate(sample_set):
        words =v.strip().split(' ')
        words = words[:min(len(words),max_num+200)]
        vectors =[]
        for j in words:
            try:
                vectors.append(word2vec_model[j])
            except:
                pass#print 'Not found %s'%j
        if len(vectors) >0:
            vectors=vectors[:min(len(vectors),max_num)]
            fishervector[i]=yael.ynumpy.fisher(gmm_model,ica.transform(np.array(vectors)).astype(np.float32) ,include='mu sigma')
    print 'mean vectors is',fishervector.mean(0)
    return fishervector
def generate_PCAmodel(sample_set,pca_model='./model/PCA/pca.model',remain_num=6000):
    pca =PCA(n_components=remain_num,copy=False)
    pca.fit(sample_set)
    joblib.dump(pca,pca_model)
def norm_func(vectors,alpha=0.5):
    sign = np.sign(vectors)
    result  = sign * (np.abs(vectors)**alpha) 
    result = (result.T/np.linalg.norm(result,axis = 1)).T
    assert np.sum(result,1)[0]-1 < 1e-5
    return result
    
def generate_feature_hdf5(file_name,text_file = './test_txt/test.txt',max_num=300):
      
    sentence = np.array([v.strip()[:max_num*5] for v in open(text_file,'r').readlines()])
    hout =h5py.File(file_name,'w')
    outdata=hout.create_dataset('feature',shape=[len(sentence),18000],dtype=np.float32)
    L=len(sentence)
    batch_size=10000
    N=np.ceil(1.0*len(sentence)/batch_size).astype(int)
    for v in range(N):
        if (v+1)*batch_size>L:
            sample_set=sentence[v*batch_size:]
            outdata[v*batch_size:,:]=generate_fishervector(sample_set,max_num=max_num)
        else:
            sample_set=sentence[v*batch_size:(v+1)*batch_size]
            outdata[v*batch_size:(v+1)*batch_size,:]=generate_fishervector(sample_set,max_num=max_num)
        print 'No.%d batch is done'%v
    hout.close() 

def putinqueue(queue,match_file):
    file_root = '/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_seg/'
    for v in match_file:
        queue.put(open(file_root+v,'r').read().strip())
def generate_feature_hdf5_from_matchlist(file_name,match_file = './trainMatching_shuffle_filter.txt',max_num=300):
    ica_model='./model/ICA/ica_ourword2vec.model'
    gmm_model_path='./model/GMM/gmm_ourword2vec.model'
    ica = joblib.load(ica_model)
    gmm_model =pkl.load(open(gmm_model_path,'r'))        
    def generate_fishervector_tmp(sample_set,max_num = 30000):
        centenrs = gmm_model[0].shape[0]
        dims = gmm_model[1].shape[1]
        fishervector = np.zeros([len(sample_set),centenrs*dims*2])+0.00001
        for i,v in enumerate(sample_set):
            words =v.strip().split(' ')
            words = words[:min(len(words),max_num+200)]
            vectors =[]
            for j in words:
                try:
                    vectors.append(word2vec_model[j])
                except:
                    pass#print 'Not found %s'%j
            if len(vectors) >0:
                vectors=vectors[:min(len(vectors),max_num)]
                fishervector[i]=yael.ynumpy.fisher(gmm_model,ica.transform(np.array(vectors)).astype(np.float32) ,include='mu sigma')
        #print 'mean vectors is',fishervector.mean(0)
        return fishervector     
    match_list = [v.strip().split('      ')[1] for v in open(match_file,'r').readlines()]  
    file_root ='/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_seg/'
    hout =h5py.File(file_name,'w')
    outdata=hout.create_dataset('feature',shape=[len(match_list),18000],dtype=np.float32)
    L=len(match_list)
    #q = Queue.Queue(maxsize=100)
    #thread =threading.Thread(target =putinqueue,args=(q,match_list))
    #thread.start()
    
    P_l=[]   
    start =time.time()
    for v in range(L):
        def compute_write(name):
            doc =open(file_root+match_list[name],'r').read().strip()#q.get()
            sample_set=[doc]
            outdata[v,:]=generate_fishervector_tmp(sample_set,max_num=max_num)
            print name,'is done'
        #print 'No.%d batch is done'%v
        P=multiprocessing.Process(target=compute_write, args=(v,))
        P.start()
        P_l.append(P)
        if v % 1000==0:
            print time.time() - start,'s to processing %d smamples'%v
    for p in P_l:
        p.join()
        print 'process done',p
    hout.close() 
def generate_all():
    generate_icamodel(train_vocabulary='./vocabulary/vocabulary_nofreq_11w.txt', 
                     model_path='./model/ICA/ica_ourword2vec_11w.model')
    generate_gmmmodel(icamodel='./model/ICA/ica_ourword2vec_11w.model', 
                     model_path='./model/GMM/gmm_ourword2vec_11w.model', 
                     train_vocabulary='./vocabulary/vocabulary_nofreq_11w.txt')
    generate_feature_hdf5(file_name='/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_sentence_fishervectors_ourword2vec_11w.h5',text_file='./train_txt/train_all.txt',max_num=800)
if __name__ == '__main__':
    #generate_gmmmodel(centersnum=60, 
                     #model_path='./model/GMM/gmm_ourword2vec_testbestcenter.model')    
    #generate_feature_hdf5_from_matchlist(file_name='/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_sentence_fishervectors_ourword2vec_11w.h5',max_num=500)  
    #generate_feature_hdf5(file_name='/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_sentence_fishervectors_ourword2vec_100words.h5',text_file='./train_txt/train_words_sort.txt',max_num=100)
    generate_feature_hdf5(file_name='./test_sentence_fishervectors_ourword2vec.h5',text_file='./test_txt/test.txt',max_num=300)
    
    
    #generate_feature_hdf5(file_name='./test_sentence_fishervectors_30norm.h5', text_file='./test_txt/test_keyword.txt', 
    #                     max_num=30)
    #sentence = np.array([v.strip() for v in open('./train_txt/train.txt').readlines()])
    #hfile =h5py.File('/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_sentence_fishervectors_ourword2vec.h5','r')
    #htrain=hfile['feature']
    #vectors = generate_fishervector(sentence[-11000:],max_num=30)
    #htrain[-11000:]=norm_func(vectors)
    #hfile.close()
   
    #a=np.zeros((18000,))
    #for v in range(120):
        #a+=hfile['feature'][v*10000:(v+1)*10000,:].mean(axis=0)
        #print v
    #np.save('/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_sentence_fishervectors_ourword2vec_mean.npy',a/120)
    #generate_PCAmodel(hread['feature'][100000:120000,:])
    #sentence = np.array([v.strip() for v in open('./train_txt/train.txt').readlines()])
    #hout =h5py.File('/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_sentence_fishervectors_norm.h5','w')
    #rdata= hread['feature']
    #outdata=hout.create_dataset('feature',shape=[rdata.shape[0],18000],dtype=np.float32)
    #L=rdata.shape[0]
    #batch_size=10000
    #N=L/batch_size
    #for v in range(N+1):
        #if (v+1)*batch_size>L:
            #vectors = rdata[v*batch_size:,:]
            #outdata[v*batch_size:,:]=norm_func(vectors)
        #else:
            #vectors = rdata[v*batch_size:(v+1)*batch_size,:]
            #outdata[v*batch_size:(v+1)*batch_size,:]=norm_func(vectors)            
        #print 'No.%d batch is done'%v
    #hout.close()
    pass