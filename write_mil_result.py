import numpy as np
import h5py

image_prob =  h5py.File('./emb/train_image_embed_mil_softmax_60000.h5','r')['embed'][:,:]
prob = np.exp(image_prob)/np.sum(np.exp(image_prob),axis=1,keepdims=True)
#im_labels= (prob >np.max(prob,axis=1,keepdims=True)*0.8).astype(int)
news_labels=np.load('./emb/train_sentence_labels.npy')
image_files = [line.strip().split()[0] for line in open('./testDummyMatching.txt').readlines()]
news_files = [line.strip().split()[1] for line in open('./testDummyMatching.txt').readlines()]
S=np.matmul(news_labels,prob.T)   
rank=np.argsort(-S,1)[:,:100]
result_list='./final_result/mil_result.txt'
fout = open(result_list, 'w')
N=len(news_labels)
for i in range(N):
    #fout.write('%d.txt'%i)
    fout.write(news_files[i])
    count =0 
    for k in range(100):
        if S[i,rank[i,k]]>0.4 or count<10:
            picId = int(rank[i,k])
            fout.write(',%s' %image_files[picId])
            count+=1
    fout.write('\n')
fout.close()
