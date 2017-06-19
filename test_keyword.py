import numpy as np
import cPickle as pkl
keywords_file = './test_txt/test.txt'#'./train_txt/train.txt'
tf= {}
tf.update([v.strip().split(' ') for v in open('./vocabulary/vocabulary_nv_4w.txt','r').readlines()])
tf['.']=1e20
x=[v.strip() for v in open(keywords_file,'r').readlines()]
instance = []
df = {}
out =open('./test_txt/test_words_sort.txt','w')
for v in x:
    words1 =v.split(' ')
    words = sorted(v.split(' '),key=lambda x:float(tf[x]))
    for w in words[:min(5,len(v))]:
        try: 
            df[w]+=1
        except:
            df[w]=1
    out.write(' '.join(words)+'\n')
x=df.items()
sort_x=sorted(x,key=lambda x:-x[1])
#voc=open('./vocabulary/keywords.txt','w')
#for v in sort_x:
#    voc.write(' '.join([v[0],str(v[1])])+'\n')
out.close()
#voc.close()