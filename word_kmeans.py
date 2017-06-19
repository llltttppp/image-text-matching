import cPickle as pkl
import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.externals import joblib
words = [v.strip().split(' ')[0] for v in open('./vocabulary/keywords.txt','r').readlines()]
word2vec_model=pkl.load(open('./model/word2vec/ourword2vec.pkl'))
X=np.array([word2vec_model[v] for v in words])
kmeans_cls = MiniBatchKMeans(n_clusters=1000, max_iter=300, 
                            batch_size=20000, 
                            n_init=5,verbose=1)
kmeans_cls.fit_transform(X)
joblib.dump(kmeans_cls,'./model/kmeans/words_cluster.model')
