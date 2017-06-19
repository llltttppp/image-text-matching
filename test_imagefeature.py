import matplotlib.pyplot as plt
import h5py
import numpy as np
import skimage.io as io
import cv2
image_file = './testDummyMatching.txt'#'./train_txt/train.txt'
fishervectors_file ='./emb/train_image_conv_20000.h5'
#'./emb/train_image_embed_GMM_ourword2vec_fakeltopK_9000cluster_30000.h5'
#'/media/ltp/40BC89ECBC89DD32/souhu_fusai/test_img_feat_3crop_mean.h5'
#'./emb/train_image_embed_GMM_ourword2vec_movemean_14000.h5'
#'./test_sentence_fishervectors_ourword2vec.h5'
#'/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_sentence_fishervectors_ourword2vec.h5'
vector_num = 19000
vector_start = 0 
docvectors = h5py.File(fishervectors_file,'r')['embed'][vector_start:vector_start+vector_num]
images = [v.strip().split(' ')[0] for v in open(image_file,'r').readlines()]
#docvectors=docvectors-docvectors.mean(0)
docvectors=docvectors/np.linalg.norm(docvectors,axis=-1,keepdims=True)

S = np.matmul(docvectors,docvectors.T)
Sort_s = -np.sort(-S)
Rank = np.argsort(-S,axis=1)[:,:10]

def print_similar_news(N):
    for v in Rank[N]:
        print images[v]
        cv2.imshow(images[v],io.imread('/media/ltp/40BC89ECBC89DD32/souhu_fusai/test/image/'+images[v]))
        cv2.waitKey(5)


def print_images(N):
    for v in Rank[N]:
        print images[v]
 