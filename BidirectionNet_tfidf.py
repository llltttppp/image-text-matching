import tensorflow as tf
import numpy as np
import h5py
import time
import sys

from test_match import test_match
class BidirectionNet:
    def __init__(self,is_training=True,is_skip=False):
        self.weight_decay = 0.0005
        self.endpoint={}
        self.is_training = is_training
	self.is_skip = is_skip
        self.keep_prob = 0.5 if is_training else 1.0
        self.build_input()
        self.build_matchnet()
        if is_training:
            self.build_summary()
    def build_input(self):
	self.lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
	img_feat_shape = [None, 512] if self.is_skip else [None, 4096]
        # positive
        self.sentence_feat = tf.placeholder(tf.float32, shape=[None,12000],name='sentence_feature')
        self.image_feat = tf.placeholder(tf.float32,shape=img_feat_shape, name='image_features')
        # negative
        self.sentence_feat_neg = tf.placeholder(tf.float32, shape=[None,12000],name='sentence_feature')
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
    
    def sentencenet(self, input_tensor, reuse=False):
        with tf.variable_scope('sentence_net', reuse=reuse) as scope:
            wd = tf.contrib.layers.l2_regularizer(self.weight_decay)
            
            s = tf.shape(input_tensor)
            sentence_fc1 = tf.contrib.layers.fully_connected(input_tensor,2048, weights_regularizer=wd, scope='s_fc1')
	    #drop_fc1 = tf.nn.dropout(sentence_fc1, self.keep_prob, name='drop_fc1')
            sentence_fc2 = tf.contrib.layers.fully_connected(sentence_fc1, 512,activation_fn=None, weights_regularizer=wd, scope='s_fc2')
	    sentence_fc2_bn = tf.contrib.layers.batch_norm(sentence_fc2, center=True, scale=True, is_training=self.is_training,
	                                                   reuse=reuse, decay=0.999, updates_collections=None, 
	                                                   scope='s_fc2_bn')
            embed = sentence_fc2_bn/tf.norm(sentence_fc2_bn,axis= -1,keep_dims=True)
        self.endpoint['sentence_fc1'] = sentence_fc1
        self.endpoint['sentence_fc2'] = embed
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
        self.sentence_fc2 = self.sentencenet(self.sentence_feat, reuse=False)
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
	    
    def build_summary(self):
        tf.summary.scalar('loss/image_center_triplet_loss', tf.reduce_mean(self.image_center_triplet_loss))
        tf.summary.scalar('loss/sentence_center_triplet_loss', tf.reduce_mean(self.sentence_center_triplet_loss))
        tf.summary.scalar('loss/reg_loss', tf.add_n(self.reg_loss))
        tf.summary.scalar('loss/total_loss', self.total_loss)
        tf.summary.scalar('misc/distance_positive', tf.reduce_mean(self.d_pos))
        tf.summary.scalar('misc/distance_negative', tf.reduce_mean(self.d_neg))
        for name, tensor in self.endpoint.items():
            tf.summary.histogram('activation/' + name, tensor)
        
        t_var = tf.trainable_variables()
	watch_list = ['s_fc1', 's_fc2']
	if not self.is_skip:
	    watch_list += ['i_fc1', 'i_fc2']
        for watch_scope in watch_list:
            weight_var = [var for var in t_var if watch_scope+'/weights' in var.name]
	    bias_var = [var for var in t_var if watch_scope+'/biases' in var.name]
            tf.summary.histogram('weights/'+watch_scope, weight_var[0])
	    tf.summary.histogram('biases/'+watch_scope, bias_var[0])
             
    def train(self, sess, maxEpoch=500, batch_size=1500, lr=0.00001, is_load=False, ckpt_path=''):
        logdir = './log/tfidf_select/run3'
	model_save_path = '/media/wwt/860G/model/tf_souhu/tfidf_select_marg05/ckpt'
        # sentence shape need to be transposed
	img_feat_file = '/media/wwt/860G/data/formalCompetition4/train_img_feat_3crop_norm1.h5'
        sentence = np.load('/media/wwt/860G/data/formalCompetition4/train_tfidf.npy').T
        h5file = h5py.File(img_feat_file, mode='r')
        image_feat_all = np.array(h5file['feature'])
	print 'image feature read from %s' %img_feat_file
        train_op = tf.train.MomentumOptimizer(self.lr, 0.9).minimize(self.total_loss)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()
        N = sentence.shape[0]
        assert N == image_feat_all.shape[0]
        batch_idx = int(N / batch_size)
        sess.run(tf.global_variables_initializer())
        if is_load:
            self.saver.restore(sess, ckpt_path)
            print '%s loaded' %ckpt_path
        step = 0
        t0 = time.time()
	print 'lr = %f' %lr
        for epoch in range(maxEpoch):
            # shuffle
            idxArr = np.random.permutation(N)
            for idx in range(batch_idx):
                interval = range(idx*batch_size , (idx+1)*batch_size)
                sentence_feat = sentence[idxArr[interval]]
                image_feat = image_feat_all[idxArr[interval]]
                # sample negative pairs
                #neg_idx = list(set(range(N)) - set(interval))
                #sample_idx = idxArr[np.array(neg_idx)[np.random.random_integers(0, N-batch_size-1, batch_size)]]
                #sentence_feat_neg = sentence[sample_idx]
                #sample_idx = idxArr[np.array(neg_idx)[np.random.random_integers(0, N-batch_size-1, batch_size)]]
                #image_feat_neg = image_feat_all[sample_idx]
		# my select function
		#image_feat, sentence_feat, image_feat_neg, sentence_feat_neg = self.select_negtive(image_feat, sentence_feat, sess)
                # train
                #feed_dict = {self.sentence_feat: sentence_feat, self.sentence_feat_neg:sentence_feat_neg,\
                #             self.image_feat: image_feat, self.image_feat_neg: image_feat_neg, self.lr: lr}
		feed_dict = {self.sentence_feat: sentence_feat, self.image_feat: image_feat, self.lr: lr}
                _, summary, total_loss = sess.run([train_op, summary_op, self.total_loss], feed_dict=feed_dict)
                
                if np.mod(step, 2) == 0:
                    summary_writer.add_summary(summary, global_step=step)
                if np.mod(step+1, 500) == 0:
                    self.saver.save(sess, model_save_path, global_step=step+1)
                if np.mod(step, 10) == 0:
                    t = (time.time() - t0)/3600
                    print '%.2f hours. Iteration %d. total loss = %.4f' %(t, step, total_loss)
		if step == 0:
		    print 'Real batch size = %d' %image_feat.shape[0]
                step += 1
	    if np.mod(epoch+1, 50)==0:
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
	    #target_tensor = self.endpoint['image_fc1']
            input_tensor = self.image_feat
            h5file = h5py.File(feat_file, 'r')
            feat_all = np.array(h5file['feature'])
            h5file.close()            
        elif scope == 'sentence':
            target_tensor = self.sentence_fc2
            input_tensor = self.sentence_feat
            feat_all = np.load(feat_file).T
        else:
            print 'invalid scope %s (must be either image or sentence)' %target_tensor.name
            sys.exit(1)   
        N = feat_all.shape[0]
        assert np.mod(N, batch_size) == 0
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
            feat_batch = feat_all[interval]
            feed_dict = {input_tensor: feat_batch}
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
    is_train = False
    is_skip = False
    is_load = True
    ckpt_path = '/media/wwt/860G/model/tf_souhu/tfidf_select/ckpt-10000' # for loading pre-trained model
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = BidirectionNet(is_training=True, is_skip=is_skip)
	if is_train:
	    model.train(sess, is_load=is_load, ckpt_path=ckpt_path)
	else:
	    # extract embedding
	    root = '/media/wwt/860G/data/formalCompetition4/'
	    model_path = '/media/wwt/860G/model/tf_souhu/tfidf_select_marg05/ckpt-8000'
	    save_path = root + 'test/val_tfidf_embed_select_marg05_8k.h5'
	    feat_file = root + 'val_tfidf.npy'#train_img_feat_3crop_norm1.h5
	    scope = 'sentence'
	    model.test_embed(sess, feat_file, model_path, scope, save_path)