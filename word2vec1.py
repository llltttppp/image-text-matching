# -*- coding:utf-8 -*- 
import word2vec
#model = word2vec.load('/media/ltp/625C80425C8012C9/word2vec/cn.cbow.bin')
#model = word2vec.WordVectors.from_binary('/media/ltp/625C80425C8012C9/word2vec/cn.cbow.bin',200)
import gensim
import tensorflow as tf
#model = gensim.models.KeyedVectors.load_word2vec_format('/media/ltp/625C80425C8012C9/word2vec/cn.cbow.bin', binary=True, unicode_errors='ignore')
qq= tf.train.string_input_producer(string_tensor=['/home/ltp/文档/机器学习课件/finalproject/tf_nopickle.npy'])
reader=tf.WholeFileReader()
key,x=reader.read(qq)
y=tf.reshape(tf.decode_raw(x, tf.float32),[-1,512])
batch = tf.train.shuffle_batch([y],5,5000,1000,enqueue_many=True,num_threads=2)
result = tf.reduce_sum(batch)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    #tf.train.batch_join(tensors_list, batch_size)
    g=sess.run([y,key,x])
    for v in range(10):
        print sess.run([result])
    print g[0],g[1]
    coord.request_stop()
    coord.join(thread)