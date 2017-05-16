import tensorflow as tf
import numpy as np
import h5py
import time
import sys
import os
from sklearn.externals import joblib

from test_match import test_match

def read_h5(file_path, dataset):
    h5file = h5py.File(file_path, 'r')
    data = np.array(h5file[dataset])
    h5file.close()
    return data
def make_if_not_exist(path):
    if not os.path.exists(path):
	os.makedirs(path)
	print 'folder %s created' %path

class BidirectionNet:
    def __init__(self,is_training=True,is_skip=False,word2vec_model='/media/wwt/860G/model/word2vec/cn.cbow.bin'):
        self.weight_decay = 0.0005
        self.endpoint={}
        self.is_training = is_training
	self.is_skip = is_skip
        self.keep_prob = 0.8 if is_training else 1.0
        self.build_input()
        self.build_matchnet()
        if is_training:
            self.build_summary()
    def build_input(self):
	tfidfDim = 20878
	T = 100
	word2vecDim = 300
	self.lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
	img_feat_shape = [None, 512] if self.is_skip else [None, 4096]
        # positive
        self.tfidf_feat = tf.placeholder(tf.float32, shape=[None,tfidfDim],name='tfidf_feature')
	self.sequence_feat = tf.placeholder(tf.float32, shape=[None, T, word2vecDim], name='sequence_feature')
	self.w2v_feat = tf.placeholder(tf.float32, shape=[None,word2vecDim],name='word2vec_feature')
	self.lda_feat = tf.placeholder(tf.float32, shape=[None,512],name='lda_feature')
        self.image_feat = tf.placeholder(tf.float32,shape=img_feat_shape, name='image_features')
        # negative
	self.tfidf_feat_neg = tf.placeholder(tf.float32, shape=[None,tfidfDim],name='tfidf_feature_negative')
	self.lda_feat_neg = tf.placeholder(tf.float32, shape=[None,512],name='lda_feature_negative')
        self.image_feat_neg = tf.placeholder(tf.float32,shape=img_feat_shape, name='image_features_negative')
    
    def bn_test(self, x, scope):
	# BN in test phase
	with tf.variable_scope(scope):
	    params_shape = [x.get_shape()[-1]]
	    beta = tf.get_variable('beta', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
	    gamma = tf.get_variable('gamma', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))	    
	    mean = tf.get_variable('moving_mean', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
	    variance = tf.get_variable('moving_variance', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
	    y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
	y.set_shape(x.get_shape())
	return y
    
    def lstm(self, input_tensor, num_units=300, reuse=False):
	with tf.variable_scope('lstm', reuse=reuse) as scope:
	    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
	    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
	    zero_state = lstm_cell.zero_state(
		        batch_size=input_tensor.get_shape()[0], dtype=tf.float32)
	    input_list = tf.unstack(input_tensor, axis=1)
	    output,_ = tf.contrib.rnn.static_rnn(lstm_cell, inputs=input_list, initial_state=zero_state)
	    lstm_output = output[-1]
	self.endpoint['LSTM_output'] = lstm_output
	return lstm_output
	    
    
    def sentencenet(self, input_tensor, reuse=False):
        with tf.variable_scope('sentence_net', reuse=reuse) as scope:
            wd = tf.contrib.layers.l2_regularizer(self.weight_decay)
            
            #lstm_embed = self.lstm(input_tensor, reuse=reuse)
            sentence_fc1 = tf.contrib.layers.fully_connected(input_tensor, 2048, weights_regularizer=wd, scope='s_fc1')
	    #drop_fc1 = tf.nn.dropout(sentence_fc1, self.keep_prob, name='drop_fc1')
            sentence_fc2 = tf.contrib.layers.fully_connected(sentence_fc1, 512,activation_fn=None, weights_regularizer=wd, scope='s_fc2')
	    sentence_fc2_bn = tf.contrib.layers.batch_norm(sentence_fc2, center=True, scale=True, is_training=self.is_training,
	                                                   reuse=reuse, decay=0.999, updates_collections=None, 
	                                                   scope='s_fc2_bn')
            embed = sentence_fc2_bn/tf.norm(sentence_fc2_bn,axis= -1,keep_dims=True)
        self.endpoint['sentence_fc1'] = sentence_fc1
        self.endpoint['sentence_fc2'] = embed
        return embed
    
    def sentence_concat(self, tfidf, lda, reuse=False):
	with tf.variable_scope('sentence_concat', reuse=reuse) as scope:
	    wd = tf.contrib.layers.l2_regularizer(self.weight_decay)
	    
	    tfidf_fc1 = tf.contrib.layers.fully_connected(tfidf, 2048, weights_regularizer=wd, scope='tfidf_fc1')	
	    lda_fc1 = tf.contrib.layers.fully_connected(lda, 64, scope='lda_fc1')
	    feat_concat = tf.concat([tfidf_fc1, lda_fc1], axis=1)
	    #drop_fc1 = tf.nn.dropout(feat_concat, self.keep_prob, name='drop_fc1')
	    sentence_fc2 = tf.contrib.layers.fully_connected(feat_concat, 512,activation_fn=None, weights_regularizer=wd, scope='s_fc2')
	    sentence_fc2_bn = tf.contrib.layers.batch_norm(sentence_fc2, center=True, scale=True, is_training=self.is_training,
		                                                   reuse=reuse, decay=0.999, updates_collections=None, 
		                                                   scope='s_fc2_bn')	
	    embed = sentence_fc2_bn/tf.norm(sentence_fc2_bn, axis= -1, keep_dims=True)
	    
	self.endpoint['tfidf_fc1'] = tfidf_fc1
	self.endpoint['lda_fc1'] = lda_fc1	
	self.endpoint['concat_embed'] = embed
	return embed
    
    def imagenet(self, image_feat, reuse=False, skip=False):
	if skip:
	    return image_feat
        with tf.variable_scope('image_net', reuse=reuse) as scope:
            wd = tf.contrib.layers.l2_regularizer(self.weight_decay)
            image_fc1 = tf.contrib.layers.fully_connected(image_feat,2048, weights_regularizer=wd,scope='i_fc1')
	    #drop_fc1 = tf.nn.dropout(image_fc1, self.keep_prob, name='drop_fc1')
            image_fc2 = tf.contrib.layers.fully_connected(image_fc1, 512, activation_fn=None, weights_regularizer=wd, scope='i_fc2')
	    image_fc2_bn = tf.contrib.layers.batch_norm(image_fc2, center=True, scale=True, is_training=self.is_training, 
	                                                reuse=reuse, decay=0.999, updates_collections=None, 
	                                                scope='i_fc2_bn')
            embed = image_fc2_bn / tf.norm(image_fc2_bn,axis=-1,keep_dims=True)
        self.endpoint['image_fc1'] = image_fc1
        self.endpoint['image_fc2'] = embed
        return embed
    def triplet_loss(self, common, pos, neg, margin=0.3):
        # d(common, pos) + margin < d(common, neg)
	self.d_pos = tf.reduce_sum(tf.squared_difference(common, pos), axis=1)
	self.d_neg = tf.reduce_sum(tf.squared_difference(common, neg), axis=1)
        return tf.reduce_sum(tf.nn.relu(self.d_pos + margin - self.d_neg))
    def positive_loss(self, x, y):
        return tf.reduce_sum(tf.squared_difference(x, y, name='positive_loss'))
    
    def build_matchnet(self):
	self.sentence_fc2 = self.sentencenet(self.w2v_feat, reuse=False)
        #self.sentence_fc2 = self.sentence_concat(self.tfidf_feat, self.lda_feat, reuse=False)
        self.image_fc2 = self.imagenet(self.image_feat, skip=self.is_skip, reuse=False)
        # compute loss
        if self.is_training:
	    # triplet loss
            #sentence_fc2_neg = self.sentencenet(self.sentence_feat_neg, reuse=True)
            #image_fc2_neg = self.imagenet(self.image_feat_neg, skip=self.is_skip, reuse=True)            
            #self.image_center_triplet_loss = self.triplet_loss(self.image_fc2, self.sentence_fc2, sentence_fc2_neg)
            #self.sentence_center_triplet_loss = self.triplet_loss(self.sentence_fc2, self.image_fc2, image_fc2_neg)
	    
	    # top k triplet loss
	    self.sentence_center_triplet_loss, self.image_center_triplet_loss = self.top_K_loss(self.sentence_fc2, self.image_fc2)
            self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	    # reg loss and total loss
            self.total_loss = tf.add_n([self.image_center_triplet_loss, self.sentence_center_triplet_loss] + self.reg_loss)
            self.saver = tf.train.Saver(max_to_keep=30)
	    self.t_var = tf.trainable_variables()
	    self.g_var = tf.global_variables()
	    self.img_var = [var for var in self.t_var if 'image' in var.name]
	    
    def build_summary(self):
        tf.summary.scalar('loss/image_center_triplet_loss', tf.reduce_mean(self.image_center_triplet_loss))
        tf.summary.scalar('loss/sentence_center_triplet_loss', tf.reduce_mean(self.sentence_center_triplet_loss))
        tf.summary.scalar('loss/reg_loss', tf.add_n(self.reg_loss))
        tf.summary.scalar('loss/total_loss', self.total_loss)
        tf.summary.scalar('misc/distance_positive', tf.reduce_mean(self.d_pos))
        tf.summary.scalar('misc/distance_negative', tf.reduce_mean(self.d_neg))
        for name, tensor in self.endpoint.items():
            tf.summary.histogram('activation/' + name, tensor)
        
        # weights
        t_var = tf.trainable_variables()
	watch_list = ['tfidf_fc1', 'lda_fc1', 's_fc1', 's_fc2']
	#watch_list = ['s_fc1', 's_fc2']
	if not self.is_skip:
	    watch_list += ['i_fc1', 'i_fc2']
        for watch_scope in watch_list:
            weight_var = [var for var in t_var if watch_scope+'/weights' in var.name]
	    bias_var = [var for var in t_var if watch_scope+'/biases' in var.name]
	    if len(weight_var) == 0:
		continue
            tf.summary.histogram('weights/'+watch_scope, weight_var[0])
	    tf.summary.histogram('biases/'+watch_scope, bias_var[0])
             
    def train(self, sess, maxEpoch=500, batch_size=1500, lr=0.00001, is_load=False, ckpt_path=''):
        logdir = './log/tfidf_w2v/run1'
	print 'log in %s' %logdir
	model_save_path = '/media/wwt/860G/model/tf_souhu/tfidf_w2v/'
	make_if_not_exist(model_save_path)
	model_save_path += 'ckpt'

	data_root = '/media/wwt/860G/data/souhu_data/fusai/'
	img_feat_file = data_root + 'train_img_feat_3crop_norm1.h5'
        w2v_file = data_root + 'train_w2v.h5'
	lda_file = data_root + '?'
        image_feat_all = read_h5(img_feat_file, 'feature')
	w2v_feat = read_h5(w2v_file, 'feature')
	N = w2v_feat.shape[0]
	assert N == image_feat_all.shape[0]	
	#lda
	#lda = read_h5(lda_file, 'feature')
	#assert N == lda.shape[0]
	print 'image feature read from %s' %img_feat_file
	step = 0
	
        train_op = tf.train.MomentumOptimizer(self.lr, 0.9).minimize(self.total_loss)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()
        batch_idx = int(N / batch_size)
	print '%d iters for each epoch' %batch_idx
        sess.run(tf.global_variables_initializer())
        if is_load:
	    loader = tf.train.Saver(var_list=self.img_var) #var_list=self.img_var
            loader.restore(sess, ckpt_path)
            print '%s loaded' %ckpt_path
	print 'step start from %d' %step
        t0 = time.time()
	print 'lr = %f' %lr
        for epoch in range(maxEpoch):
            # shuffle
            idxArr = np.random.permutation(N)
            for idx in range(batch_idx):
                interval = range(idx*batch_size , (idx+1)*batch_size)
                sentence_feat = w2v_feat[idxArr[interval]]
                image_feat = image_feat_all[idxArr[interval]]
		#lda_feat = lda[idxArr[interval]]

                # train
		#feed_dict = {self.tfidf_feat: tfidf_feat, self.lda_feat: lda_feat, self.image_feat: image_feat, self.lr: lr}
		feed_dict = {self.w2v_feat: sentence_feat, self.image_feat: image_feat, self.lr: lr}
		
                _, summary, total_loss = sess.run([train_op, summary_op, self.total_loss], feed_dict=feed_dict)
                
                if np.mod(step, 5) == 0:
                    summary_writer.add_summary(summary, global_step=step)
                if np.mod(step+1, 500) == 0:
                    self.saver.save(sess, model_save_path, global_step=step+1)
		    print 'model saved to %s' %model_save_path
                if np.mod(step, 10) == 0:
                    t = (time.time() - t0)/3600
                    print '%.2f hours. Iteration %d. total loss = %.4f' %(t, step, total_loss)
                step += 1
	    if np.mod(epoch+1, 400)==0:
		lr *= 0.1
		print 'lr scaled to %f' %lr
		
    def train_multiDataset(self, sess, maxEpoch=500, batch_size=4000, lr=0.00001, is_load=False, ckpt_path=''):
        logdir = './log/tfidf_w2v/run2'
	print 'log in %s' %logdir
	model_save_path = '/home/litp/souhu/model/tf_souhu/tfidf_w2v/'
	make_if_not_exist(model_save_path)
	model_save_path += 'ckpt'
	data_root = '/home/litp/souhu_data/fusai/'
	img_feat_file = data_root + 'train_img_feat_3crop_mean.h5'
	dataset_size = 249900
	w2v_file = data_root + '/train/train_tfidf_weighted_word2vec.h5'	
	
	print 'image feature read from %s' %img_feat_file	
	w2v_feat_all = read_h5(w2v_file, 'feature')
	img_h5 = h5py.File(img_feat_file, 'r')
	nDataset = len(img_h5.keys())
	step = 0
	train_op = tf.train.MomentumOptimizer(self.lr, 0.9).minimize(self.total_loss)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
	
        if is_load:
	    loader = tf.train.Saver(var_list=self.img_var) #var_list=self.img_var
            loader.restore(sess, ckpt_path)
            print '%s loaded' %ckpt_path
	print 'step start from %d' %step
	
        t0 = time.time()
	print 'lr = %f. Multidataset with %d sets. run over one set in %d iters' %(lr, nDataset, int(dataset_size/batch_size))
	print 'w2v_feat length = %d' %w2v_feat_all.shape[0]
        for epoch in range(maxEpoch):
	    for setId in range(nDataset):
		shift = setId * dataset_size
		img_feat_all = np.array(img_h5['feature_'+str(setId)], dtype=np.float32)
		N = img_feat_all.shape[0]
		if setId < nDataset - 1:
		    assert N == dataset_size
		#    w2v_feat = w2v_feat_all[setId * dataset_size : (setId + 1) * dataset_size, :]
		#else:
		#    w2v_feat = w2v_feat_all[setId * dataset_size : , :]   
		# shuffle
		idxArr = np.random.permutation(N)
		batch_idx = int(N / batch_size)
		for idx in range(batch_idx):
		    interval = range(idx*batch_size , (idx+1)*batch_size)
		    sentence_feat = w2v_feat_all[idxArr[interval] + shift]
		    image_feat = img_feat_all[idxArr[interval]]
		    #lda_feat = lda[idxArr[interval]]
    
		    # train
		    #feed_dict = {self.tfidf_feat: tfidf_feat, self.lda_feat: lda_feat, self.image_feat: image_feat, self.lr: lr}
		    feed_dict = {self.w2v_feat: sentence_feat, self.image_feat: image_feat, self.lr: lr}
		    
		    _, summary, total_loss = sess.run([train_op, summary_op, self.total_loss], feed_dict=feed_dict)
		    
		    if np.mod(step, 5) == 0:
			summary_writer.add_summary(summary, global_step=step)
		    if np.mod(step+1, 500) == 0:
			self.saver.save(sess, model_save_path, global_step=step+1)
			print 'model saved to %s' %model_save_path
		    if np.mod(step, 10) == 0:
			t = (time.time() - t0)/3600
			print '%.2f hours. Iteration %d. total loss = %.4f' %(t, step, total_loss)
		    step += 1
	    if np.mod(epoch+1, 400)==0:
		lr *= 0.1
		print 'lr scaled to %f' %lr    
    
    def select_negtive(self, i_feat, s_feat, sess, topN=50):
	'''
	Select the triplets with the largest losses \n
	return i_feat_pos, s_feat_pos, i_feat_neg, s_feat_neg
	'''
	feed_dict = {self.image_feat: i_feat, self.sentence_feat:s_feat}
	i_embed, s_embed = sess.run([self.image_fc2, self.sentence_fc2], feed_dict=feed_dict)
	S = np.matmul(i_embed, s_embed.T)
	i_feat_pos = i_feat.repeat(topN, axis=0)
	s_feat_pos = s_feat.repeat(topN, axis=0)
	N = S.shape[0]
	np.fill_diagonal(S, -2*np.ones(N))
	neg_s_idx = S.argsort(axis=1)[:, -topN:]
	neg_i_idx = S.argsort(axis=0)[-topN:, :]
	s_feat_neg = s_feat[neg_s_idx.flatten('C')]
	i_feat_neg = i_feat[neg_i_idx.flatten('F')]
	return i_feat_pos, s_feat_pos, i_feat_neg, s_feat_neg
    
    def top_K_loss(self, sentence, image, K=30, margin=0.5):
	sim_matrix = tf.matmul(sentence, image, transpose_b=True)
	s_square = tf.reduce_sum(tf.square(sentence), axis=1)
	im_square = tf.reduce_sum(tf.square(image), axis=1)
	d = tf.reshape(s_square,[-1,1]) - 2 * sim_matrix + tf.reshape(im_square, [1, -1])
	positive = tf.stack([tf.matrix_diag_part(d)] * K, axis=1)
	length = tf.shape(d)[-1]
	d = tf.matrix_set_diag(d, 8 * tf.ones([length]))
	sen_loss_K ,_ = tf.nn.top_k(-1.0 * d, K, sorted=False) # note: this is negative value
	im_loss_K,_ = tf.nn.top_k(tf.transpose(-1.0 * d), K, sorted=False) # note: this is negative value
	sentence_center_loss = tf.nn.relu(positive + sen_loss_K + margin)
	image_center_loss = tf.nn.relu(positive + im_loss_K + margin)
	self.d_neg = (sen_loss_K + im_loss_K)/-2.0
	self.d_pos = positive
	self.endpoint['debug/im_loss_topK'] = -1.0 * im_loss_K
	self.endpoint['debug/sen_loss_topK'] = -1.0 * sen_loss_K 
	self.endpoint['debug/d_Matrix'] = d
	self.endpoint['debug/positive'] = positive
	self.endpoint['debug/s_center_loss'] = sentence_center_loss
	self.endpoint['debug/i_center_loss'] = image_center_loss
	self.endpoint['debug/S'] = sim_matrix
	self.endpoint['debug/sentence_square'] = s_square
	self.endpoint['debug/image_square'] = im_square
	return tf.reduce_sum(sentence_center_loss), tf.reduce_sum(image_center_loss)    
    
    def test_embed(self, sess, feat_file, model_path, scope, save_path, h5dataset='embed', batch_size=100):
        '''
        For testing. Generate the final embedding of images for image-text matching.
        Dataset: 'embed'
        Scope: either 'image' or 'sentence'
        ''' 
        # read input features
        if scope == 'image':
            target_tensor = self.image_fc2
            input_tensor = self.image_feat
            h5file = h5py.File(feat_file, 'r')
            feat_all = np.array(h5file['feature'])
            h5file.close()     
	    N = feat_all.shape[0]
	    assert np.mod(N, batch_size) == 0	    
        elif scope == 'sentence':
	    lda_file = '/media/wwt/860G/data/formalCompetition4/val_fusai_lda.h5'
            target_tensor = self.sentence_fc2
            tfidf_feat = np.load(feat_file).T
	    h5file = h5py.File(lda_file, 'r')
	    lda_feat = np.array(h5file['feature'])
	    h5file.close()
	    N  = tfidf_feat.shape[0]
	    assert  N == lda_feat.shape[0]
	    print 'using lda file %s' %lda_file
        else:
            print 'invalid scope %s (must be either image or sentence)' %target_tensor.name
            sys.exit(1)   
	#sess.run(tf.global_variables_initializer())
        # load model
        g_var = tf.global_variables()
        #load_var = [var for var in g_var if scope in var.name]
        loader = tf.train.Saver()
        loader.restore(sess, model_path)
        # forward
        embed = []
        t0 = time.time()
        for idx in range(N/batch_size):
            interval = np.array(range(idx * batch_size, (idx + 1) * batch_size))
	    if scope == 'image':
		feat_batch = feat_all[interval]
		feed_dict = {input_tensor: feat_batch}
	    else:
		tfidf_batch = tfidf_feat[interval]
		#lda_batch = lda_feat[interval]
		feed_dict = {self.tfidf_feat: tfidf_batch}
		#feed_dict = {self.tfidf_feat: tfidf_batch, self.lda_feat:lda_batch}
            embed_batch = sess.run(target_tensor, feed_dict=feed_dict)
            embed.extend(embed_batch.copy())
            if np.mod(idx, 10) == 0:
                t = (time.time() - t0)/60
                print '%.2f minutes. Iteration %d/%d' %(t, idx, N/batch_size)
        # save embed
        embed = np.array(embed, dtype=np.float32)
        h5file = h5py.File(save_path, 'w')
        h5file.create_dataset(h5dataset, data=embed, dtype=np.float32)
        h5file.close()
	print 'target tensor %s' %target_tensor.op.name
        print 'embed done for scope %s. Saved shape ' %scope, embed.shape
	print 'saved to %s' %save_path

if __name__ == '__main__':
    is_train = True
    is_skip = False
    is_load = True
    ckpt_path = '/home/litp/souhu/model/tf_souhu/tfidf_w2v/ckpt-5000' # for loading pre-trained model
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = BidirectionNet(is_training=True, is_skip=is_skip)
	if is_train:
	    model.train_multiDataset(sess, is_load=is_load, ckpt_path=ckpt_path)
	else:
	    # extract embedding
	    root = '/media/wwt/860G/data/formalCompetition4/'
	    model_path = '/media/wwt/860G/model/tf_souhu/concat2w/ckpt-2000'
 	    save_path = root + 'test/val_fusai_img_embed_tfidf_2k.h5'
	    feat_file = root + 'val_fusai_img_feat_3crop_norm1.h5'#train_img_feat_3crop_norm1.h5
	    scope = 'image'
	    model.test_embed(sess, feat_file, model_path, scope, save_path)
