import tensorflow as tf
import numpy as np
import h5py
import time
import sys
import os
instance_list ={}
instance_list.update([(v.strip().split(' ')[0],int(v.strip().split(' ')[1])) for v in open('./vocabulary/keywords_labels.txt','r').readlines()])
prob = h5py.File('./emb/train_image_embed_mil_softmax_60000.h5','r')['embed'][:,:]
image_list =[v.strip().split(' ')[0] for v in open('testDummyMatching.txt','r').readlines()]
def show_instance(num):
    rank =np.argsort(-prob[num])[:1]
    print image_list[num]
    count=0
    for v in instance_list.items():
        if v[1] in rank:
            print v[0]
            count+=1
    print count
                         
