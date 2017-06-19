import preprocessing
import h5py
import numpy as np
news_file = './test_txt/test_cont.txt'#'./train_txt/train.txt'
fishervectors_file ='./emb/train_sentence_embed_nconv_30000.h5'
#'/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_sentence_fishervectors_ourword2vec.h5'
#'./emb/train_sentence_embed_GMM_ourword2vec_movemean_14000.h5'
#'./test_sentence_fishervectors_ourword2vec.h5'
#'/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_sentence_fishervectors_ourword2vec.h5'
vector_num = 19000
vector_start = 0 
docvectors = h5py.File(fishervectors_file,'r')['embed'][vector_start:vector_start+vector_num]
docvectors = preprocessing.norm_func(vectors=docvectors,alpha=1.0)
news = [v.strip() for v in open(news_file,'r').readlines()]

S = np.matmul(docvectors,docvectors.T)
Sort_S = -np.sort(-S)
Rank = np.argsort(-S,axis=1)[:,:15]

def print_similar_news(N):
    for v in Rank[N]:
        print news[v]
