# -*- coding:utf-8 -*- 
import word2vec
#model = word2vec.load('/media/ltp/625C80425C8012C9/word2vec/cn.cbow.bin')
#model = word2vec.WordVectors.from_binary('/media/ltp/625C80425C8012C9/word2vec/cn.cbow.bin',200)
import gensim
import tensorflow as tf
import tflearn.data_flow as df
import h5py
#model = gensim.models.KeyedVectors.load_word2vec_format('/media/ltp/625C80425C8012C9/word2vec/cn.cbow.bin', binary=True, unicode_errors='ignore')
#qq= tf.train.string_input_producer(string_tensor=['/home/ltp/文档/机器学习课件/finalproject/tf_nopickle.npy'])
#reader=tf.WholeFileReader()
#key,x=reader.read(qq)
#y=tf.reshape(tf.decode_raw(x, tf.float32),[-1,512])
#batch = tf.train.shuffle_batch([y],5,5000,1000,enqueue_many=True,num_threads=2)
#result = tf.reduce_sum(batch)
X=tf.placeholder(tf.float32)
Z=tf.reduce_mean(X)
with tf.Session() as sess:
    hread =h5py.File('/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_sentence_fishervectors.h5','r')
    coord = tf.train.Coordinator()
    G=df.FeedDictFlow(feed_dict={X:hread}, coord=coord,batch_size=1,num_threads=2)
    G.start()
    for v in range(1):
        x=G.next()#sess.run(Z,feed_dict = G.next())
    G.stop()
    #thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    ##tf.train.batch_join(tensors_list, batch_size)
    #g=sess.run([y,key,x])
    #for v in range(10):
        #print sess.run([result])
    #print g[0],g[1]
    #coord.request_stop()
    #coord.join(thread)