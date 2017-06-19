import h5py
import numpy as np
data_root = './final_result/'

result_file= './final_result/GMM_esemble_top1.txt'
result_list =['tfidf_1w.txt','tfidf5_7k.txt','GMM_ourword2vec_norm_10000.txt','GMM1_29500.txt','GMM_norm_margin0.1_17500.txt',
              'GMM1_earlystop_16000.txt','GMM_ourword2vec_movemean_8500.txt','tfidf2_1w.txt','tfidf4_2w.txt','tfidf3_1w2.txt',
              'GMM_ourword2vec_wwttopK9000feat_8500.txt',
              'tfidf_rand1_2w7.txt','tfidf_rand2_1w8.txt',
              #'GMM_ourword2vec_fakeltopK_9000cluster_10000.txt',
              #'GMM_ourword2vec_9000cluster_20000.txt'
              ]
#result_list=['tfidf_1w.txt','GMM_ourword2vec_norm_10000.txt','GMM1_29500.txt',
              #'GMM1_earlystop_16000.txt','GMM_ourword2vec_movemean_8500.txt','tfidf2_1w.txt','tfidf4_2w.txt','tfidf3_1w2.txt',
              #'GMM_ourword2vec_new_topK_15000.txt','GMM_ourword2vec_wwttopK9000feat_8500.txt',
              #'tfidf_rand1_2w7.txt','tfidf_rand2_1w8.txt',
              #'GMM_ourword2vec_9000cluster_20000.txt']
#result_list=['tfidf2_1w.txt','tfidf3_1w2.txt','tfidf4_2w.txt','tfidf5_7k.txt','tfidf6_1w4.txt','tfidf_1w.txt']
file_list=[]
res_list=[]
fout = open(result_file, 'w')
for v in result_list:
    file_list.append(np.array(open(data_root+v,'r').readlines()))
L=19997
for v in range(L):
    Z ={}
    
    num={}
 
    text=''
    for f in file_list:
        s=f[v].strip().split(',')
        text =s[0]
        for i,image in enumerate(s[1:]):
            try:
                Z[image]+=i
                num[image]+=1
            except:
                Z[image]=i
                num[image]=1
    res = sorted(Z.keys(),key=lambda x:(Z[x]+10000.0*(len(result_list)-num[x]))/len(
                                                                             result_list))
    res_list.append([(v,(Z[v]+10000.0*(len(result_list)-num[v]))/len(
                                                                             result_list) )for v in res])  
    fout.write(text)
    for im in res[:10]:
        fout.write(','+res[0])
        #fout.write(','+im)
        
    fout.write('\n')
fout.close()
    
            

