import numpy as np
import h5py
x=h5py.File('./emb/train_sentence_embed_GMM_ourword2vec_softmax_loss_17000.h5','r')
s_f=np.array(x['embed'])
#x=h5py.File('./emb/validate_end_sentence_embed_pca_simpleconv_crop3_fewwords_ftlm11500.h5','r')
#s_end_f=np.array(x['embed'])
x=h5py.File('./emb/train_image_embed_GMM_ourword2vec_softmax_loss_17000.h5','r')
image_f=np.array(x['embed'])

image_files = [line.strip().split()[0] for line in open('./testDummyMatching.txt').readlines()]
news_files = [line.strip().split()[1] for line in open('./testDummyMatching.txt').readlines()]
#S=s_f.dot(image_f.T)
l = []
for v in range(len(s_f)):
    print v
    l.append(np.sum(np.abs(s_f-image_f[v,:]),axis=1))
S=np.stack(l,axis=0)
rank=np.argsort(S,1)[:,:10]
result_list='./result/GMM_ourword2vec_softmax_loss_17000.txt'
#image_files = [line.strip() for line in open('./val_image.txt').readlines()]





fout = open(result_list, 'w')
N=len(image_files)
for i in range(N):
    #fout.write('%d.txt'%i)
    fout.write(news_files[i])
    for k in range(10):
        picId = int(rank[i,k])
        fout.write(',%s' %image_files[picId])
    fout.write('\n')
fout.close()