import cPickle as pkl
import numpy as np
instance_list ={}
instance_list.update([(v.strip().split(' ')[0],int(v.strip().split(' ')[1])) for v in open('./vocabulary/keywords_labels.txt','r').readlines()])      
x=np.zeros((1000,))
count=0
for v in open('./train_txt/train.txt').readlines():
    words = v.strip().split(' ')
    tmp=np.zeros((1000,))
    for w in words:
        try:
            tmp[instance_list[w]]=1
        except:
            pass
    count+=1
    x+=tmp
    if count%1000==0:
        print count