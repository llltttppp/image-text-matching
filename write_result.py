import numpy as np
import h5py
#emb_list=['./emb/train_image_embed_GMM1_29500.h5','./emb/train_sentence_embed_GMM1_29500.h5',
            #'./emb/train_image_embed_GMM1_earlystop_16000.h5','./emb/train_sentence_embed_GMM1_earlystop_16000.h5',
            #'./emb/train_image_embed_GMM_ourword2vec_movemean_8500.h5','./emb/train_sentence_embed_GMM_ourword2vec_movemean_8500.h5',
            #'./emb/train_image_embed_GMM_ourword2vec_wwttopK9000feat_8500.h5','./emb/train_sentence_embed_GMM_ourword2vec_wwttopK9000feat_8500.h5',
             #'./emb/train_image_embed_GMM_ourword2vec_9000cluster_20000.h5','./emb/train_sentence_embed_GMM_ourword2vec_9000cluster_20000.h5',
        #'./emb/train_image_embed_GMM_ourword2vec_faketopK_9000cluster_10000.h5','./emb/train_sentence_embed_GMM_ourword2vec_faketopK_9000cluster_10000.h5','./emb/train_image_embed_GMM_ourword2vec_norm_10000.h5','./emb/train_sentence_embed_GMM_ourword2vec_norm_10000.h5',
                    #'./emb/train_image_embed_GMM_norm_margin0.1_17500.h5','./emb/train_sentence_embed_GMM_norm_margin0.1_17500.h5']


#result_namelist=['GMM1_29500.txt','GMM1_earlystop_16000.txt','GMM_ourword2vec_movemean_8500.txt','GMM_ourword2vec_wwttopK9000feat_8500.txt',
                 #'GMM_ourword2vec_9000cluster_20000.txt','GMM_ourword2vec_fakeltopK_9000cluster_10000.txt',
                 #'GMM_ourword2vec_norm_10000.txt','GMM_norm_margin0.1_17500.txt',            
                                          #'tfidf_1w.txt','tfidf5_7k.txt','tfidf2_1w.txt','tfidf4_2w.txt','tfidf3_1w2.txt',
              #'tfidf_rand1_2w7.txt','tfidf_rand2_1w8.txt']
emb_list=['./emb/train_image_embed_GMM_ourword2vec_9000feat_softmax_8500.h5','./emb/train_sentence_embed_GMM_ourword2vec_9000feat_softmax_8500.h5']   
result_namelist=['GMM_ourword2vec_9000feat_softmax_8500.txt']
for v in range(1):
    x=h5py.File(emb_list[v*2+1],'r')
    s_f=np.array(x['embed'])
    #x=h5py.File('./emb/validate_end_sentence_embed_pca_simpleconv_crop3_fewwords_ftlm11500.h5','r')
    #s_end_f=np.array(x['embed'])
    x=h5py.File(emb_list[v*2],'r')
    image_f=np.array(x['embed'])
    
    image_files = [line.strip().split()[0] for line in open('./testDummyMatching.txt').readlines()]
    news_files = [line.strip().split()[1] for line in open('./testDummyMatching.txt').readlines()]
    #S=s_f.dot(image_f.T)
    S=np.matmul(s_f,image_f.T)
    #S2=np.matmul(image_f,s_f.T)
    #im_S =np.matmul(image_f,image_f.T)
    rank=np.argsort(-S,1)[:,:10]
    #rank2 = np.argsort(-S2,1)[:,:10]
    #result_list='./result/'+result_namelist[v]
    result_list='./final_result/'+result_namelist[v]
    #result_list='./result/GMM_ourword2vec_fakeltopK_9000cluster_10000.txt'
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