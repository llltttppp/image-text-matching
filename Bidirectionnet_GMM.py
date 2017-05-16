# -*- coding:utf-8 -*- 
import numpy as np
import h5py
import time
import os
import sys
import cPickle as pkl
import tensorflow as tf
slim = tf.contrib.slim

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print 'folder %s created' %path
class BidirectionNet:
    def __init__(self,is_training=True,is_skip=False,is_TopKloss=True,word2vec_model='./model/word2vec/word2vec_11w.pkl',batch_size=500,is_keep_prob=False):
        self.word2vec = pkl.load(open(word2vec_model,'r'))
        self.batch_size = batch_size
        self.weight_decay = 0.0001
        self.endpoint={}
        self.is_skip=is_skip
        self.is_TopKloss = is_TopKloss
        self.is_training = is_training
        self.keep_prob = 0.5 if is_keep_prob else 1.0
        self.build_input()
        self.build_matchnet()
        if is_training:
            self.build_summary()
    def build_input(self):
        # positive
        self.raw_sentence= tf.placeholder(tf.float32, shape=[self.batch_size,18000],name='raw_sentence')
        self.sentence_emb =self.raw_sentence #tf.nn.embedding_lookup(tf.get_variable('word_embedding',[4096,512]),self.raw_sentence)
        self.image_feat = tf.placeholder(tf.float32,shape=[self.batch_size,4096], name='image_features')   
    def conv_layer(self, X, num_output, kernel_size, s, p='SAME'):
        return tf.contrib.layers.conv2d(X,num_output,kernel_size,s,\
                                        padding=p,weights_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),\
                                        normalizer_fn=tf.contrib.layers.batch_norm,normalizer_params={'is_training':self.is_training,'updates_collections':None})
    def sentencenet(self, sentence_emb, reuse=False):
        with tf.variable_scope('sentence_net', reuse=reuse) as scope:
            wd = tf.contrib.layers.l2_regularizer(self.weight_decay)
            sentence_fc1 =tf.contrib.layers.fully_connected(self.raw_sentence,2048, \
                                                            weights_regularizer=wd, scope='s_fc1') # 20*10*256
            sentence_fc2 = tf.contrib.layers.fully_connected(sentence_fc1, 512,activation_fn=None,normalizer_fn=tf.contrib.layers.batch_norm,\
                                                             normalizer_params={'is_training':self.is_training,'updates_collections':None}, weights_regularizer=wd, scope='s_fc2')
            sentence_fc2 = sentence_fc2/tf.norm(sentence_fc2,axis= -1,keep_dims=True)
        self.endpoint['sentence_fc1'] = sentence_fc1
        self.endpoint['sentence_fc2'] = sentence_fc2
        return sentence_fc2
    def imagenet(self, image_feat, reuse=False,skip=False):
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
    def triplet_loss(self, common, pos, neg, margin=0.2):
        # d(common, pos) + margin < d(common, neg)
        self.d_pos = tf.reduce_sum(tf.squared_difference(common, pos),-1)
        self.d_neg =tf.reduce_sum(tf.squared_difference(common, neg),-1)
        return tf.reduce_sum(tf.nn.relu(self.d_pos + margin - self.d_neg, name = 'triplet_loss'))

    def build_matchnet(self):
        self.sentence_fc2 = self.sentencenet(self.sentence_emb, reuse=False)
        self.image_fc2 = self.imagenet(self.image_feat, reuse=False,skip=self.is_skip)
        # compute loss
        if self.is_training:

            self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.positiveloss=self.positive_loss(self.sentence_fc2,self.image_fc2)
            if not self.is_TopKloss:
                self.total_loss=tf.add_n([self.positive_loss(self.sentence_fc2,self.image_fc2)]+self.reg_loss)
            else:            
                self.total_loss =tf.add_n( list(self.top_K_loss(self.sentence_fc2,self.image_fc2))+self.reg_loss)
            self.saver = tf.train.Saver(max_to_keep=20)
    def build_summary(self):
        tf.summary.scalar('loss/reg_loss', tf.add_n(self.reg_loss))
        tf.summary.scalar('loss/positive_loss', self.positiveloss)
        tf.summary.scalar('loss/total_loss', self.total_loss)
        if self.is_skip:
            tf.summary.histogram('activation/image_fc2',self.image_fc2)
        if self.is_TopKloss:
            tf.summary.scalar('msic/dneg', self.d_neg)
            tf.summary.scalar('msic/dpos', self.d_pos)        
        for name, tensor in self.endpoint.items():
            tf.summary.histogram('activation/' + name, tensor)

        t_var = tf.trainable_variables()
        watch_list = ['s_fc1', 's_fc2']
        if not self.is_skip:
            watch_list += ['i_fc1', 'i_fc2']        
        for watch_scope in watch_list:
            watch_var = [var for var in t_var if watch_scope+'/weights' in var.name]
            tf.summary.histogram('weights/'+watch_scope, watch_var[0])
    def positive_loss(self, sentence, image):
        diff = tf.reduce_sum(tf.squared_difference(sentence, image, name='positive_loss')) 
        return diff     
    def top_K_loss(self,sentence,image,K=50,margin=0.4):
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
        self.d_neg = tf.reduce_mean((sen_loss_K + im_loss_K)/-2.0)
        self.d_pos =tf.reduce_mean(positive)
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
    def build_trainop(self,loss,lr=0.001,clipping_norm=10,optimizer =tf.train.AdadeltaOptimizer,tvars=None,clip_vars=None):
        if tvars is None:        
            tvars = tf.trainable_variables()
        if clip_vars is None:
            clip_vars = tvars
        g=tf.gradients(loss, tvars)
        grads= [tf.clip_by_global_norm(v,clipping_norm) if v in clip_vars else v for v in g ]
        opt = optimizer(lr)
        for i,v in enumerate(tvars):
            tf.summary.histogram(name=v.name+'_gradient', values=grads[i])
        return opt.apply_gradients(zip(grads,tvars))   

    def train(self, sess, maxEpoch=300, lr=0.0001,is_load=False,ckpt_path=''):
        logdir = './log/run_Bidirectionnet_GMM/'
        model_save_path='./model/Bidirectionnet_lstm/'
        make_if_not_exist(model_save_path)
        model_save_path+='model'
        data_root = '/media/ltp/40BC89ECBC89DD32/souhu_fusai/'
        sentence = h5py.File(data_root+'train_sentence_fishervectors.h5',mode='r')
        h5file = h5py.File(data_root+'train_img_feat_3crop_mean_all.h5', mode='r')
        image_feat_all = h5file['feature']
        train_op =self.build_trainop(self.total_loss,lr=lr,clipping_norm=100000)

        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()
        N = image_feat_all.shape[0]
        batch_idx = int(N / self.batch_size)
        sess.run(tf.global_variables_initializer())
        if is_load:
            self.saver.restore(sess, ckpt_path)
            print '%s loaded' %ckpt_path   
            
                


        step =0
        t0 = time.time()
        for epoch in range(maxEpoch):
            # shuffle
            assert N==sentence.shape[0]
            idxArr = np.random.permutation(N)
            for idx in range(batch_idx):
                interval = range(idx*self.batch_size , (idx+1)*self.batch_size)
                raw_sentence = self.read_wordvector(sentence[idxArr[interval]],self.batch_size)
                image_feat = image_feat_all[idxArr[interval]]
                # sample
                # train
                feed_dict = {self.raw_sentence: raw_sentence, self.image_feat: image_feat}
                _, summary, total_loss = sess.run([train_op, summary_op, self.total_loss], feed_dict=feed_dict)

                if np.mod(step, 1) == 0:
                    summary_writer.add_summary(summary, global_step=step)
                if np.mod(step, 500) == 0:
                    self.saver.save(sess, model_save_path, global_step=step)
                if np.mod(step, 100) == 0:
                    t = (time.time() - t0)/3600
                    print '%.2f hours. Iteration %d. total loss = %.4f' %(t, step, total_loss)
                step += 1

    def train_multidataset(self, sess, maxEpoch=300, lr=0.0001,is_load=False,ckpt_path='',only_image = False):
        logdir = './log/Bidirectionnet_GMM/GMM1/'
        print 'log in %s' %logdir
        model_save_path = './model/GMM/GMM1/'
        make_if_not_exist(model_save_path)
        model_save_path += 'model'
        data_root = '/media/ltp/40BC89ECBC89DD32/souhu_fusai/'
        img_feat_file = data_root + 'train_img_feat_3crop_mean_all.h5'
        
        sentence_feat_file =data_root+'train_sentence_fishervectors.h5'
    
        print 'image feature read from %s' %img_feat_file
        print 'sentence feature read from %s' %sentence_feat_file
        img_h5 = h5py.File(img_feat_file, 'r')
        sen_h5 = h5py.File(sentence_feat_file, 'r')
        L_im = img_h5['feature'].shape[0]
        L_sen = sen_h5['feature'].shape[0]
        assert L_im == L_sen
        nDataset = 6
        dataset_size = int(L_im/nDataset)        
        step = 0
        train_op =self.build_trainop(self.total_loss,lr=lr,clipping_norm=500)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
    
        if is_load:
            self.saver.restore(sess, ckpt_path) #var_list=self.img_var
            print '%s loaded' %ckpt_path
        print 'step start from %d' %step
    
        t0 = time.time()
        for epoch in range(maxEpoch):
            for setId in np.random.permutation(nDataset):
                shift = setId * dataset_size
                
                if setId < nDataset - 1:
                    img_feat_all = img_h5['feature'][shift:shift+dataset_size,:]
                    raw_sentence = sen_h5['feature'][shift:shift+dataset_size,:]
                    N = img_feat_all.shape[0]                    
                    assert N == dataset_size
                else:
                    img_feat_all = img_h5['feature'][shift:,:]
                    raw_sentence = sen_h5['feature'][shift:,:]
                    N = img_feat_all.shape[0] 
                for _ in range(2):
                    idxArr = np.random.permutation(N)
                    batch_idx = int(N / self.batch_size)
                    for idx in range(batch_idx):
                        interval = range(idx*self.batch_size , (idx+1)*self.batch_size)
                        raw_sentence = self.read_wordvector(sentence[idxArr[interval]],self.batch_size)
                        image_feat = img_feat_all[idxArr[interval]]
                        #lda_feat = lda[idxArr[interval]]
        
                        # train
                        #feed_dict = {self.tfidf_feat: tfidf_feat, self.lda_feat: lda_feat, self.image_feat: image_feat, self.lr: lr}
                        feed_dict = {self.raw_sentence:raw_sentence, self.image_feat: image_feat}
        
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

    def test_embed(self, sess, feat_file, model_path, scope, save_path, h5dataset='embed'):
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
        elif scope == 'sentence':
            target_tensor = self.sentence_fc2
            input_tensor = self.raw_sentence
            feat_all = np.array(open(feat_file,'r').readlines())
        else:
            print 'invalid scope %s (must be either image or sentence)' %target_tensor.name
            sys.exit(1)   
        #feat_all = feat_all[:10000, ...]
        N = feat_all.shape[0]
        assert np.mod(N, self.batch_size) == 0
        # load model
        t_var = tf.global_variables()
        load_var = [var for var in t_var ]
        loader = tf.train.Saver(var_list = load_var)
        loader.restore(sess, model_path)
        # forward
        embed = []
        t0 = time.time()
        for idx in range(N/self.batch_size):
            interval = np.array(range(idx * self.batch_size, (idx + 1) * self.batch_size))
            feat_batch = feat_all[interval]
            if scope =='sentence':
                feat_batch =self.read_wordvector(feat_batch,self.batch_size)
            feed_dict = {input_tensor: feat_batch}
            embed_batch = sess.run(target_tensor, feed_dict=feed_dict)
            embed.extend(embed_batch)
            if np.mod(idx, 10) == 0:
                t = (time.time() - t0)/60
                print '%.2f minutes. Iteration %d/%d' %(t, idx, N/self.batch_size)
        # save embed
        embed = np.array(embed)
        h5file = h5py.File(save_path, 'w')
        h5file.create_dataset(h5dataset, data=embed)
        h5file.close()
        print 'embed done for scope %s. Saved shape ' %scope, embed.shape


    def read_wordvector(self,batch_sentences,batch_size):
        #ss=time.time()
        batch_vectors=np.zeros([batch_size,self.sentence_len,self.vector_size])
        for i,v in enumerate(batch_sentences):
            sentence_matrix = np.zeros([self.sentence_len,self.vector_size])
            vsp = v.strip().split(' ')
            assert len(vsp) >= self.sentence_len
            for j,word in enumerate(vsp):
                if j>= self.sentence_len:
                    break                
                try:
                    wordvec = self.word2vec[str(word)]
                    sentence_matrix[j,:]=wordvec
                except:
                    print 'ignore %s'%word
                
            batch_vectors[i,:,:]=sentence_matrix
        #print time.time()-ss
        return batch_vectors
        

    
if __name__ == '__main__':
    is_train = True
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        if is_train:
            model = BidirectionNet(is_training=True,is_skip=False,is_TopKloss =True,batch_size=2000,is_keep_prob=False)
            #model.train(sess,lr=0.0001,is_load=True,ckpt_path='./model/Bidirectionnet_lstm/model-1500')
            model.train_multidataset(sess,lr=0.001,is_load=True,ckpt_path='./model/Bidirectionnet_lstm/fc6000/model-2000')
            #model.train(sess,lr=0.0001)
        else:
            model = BidirectionNet(is_training=True,is_skip=False,is_TopKloss =True,batch_size=149,is_keep_prob=False)
            feat_file = './train_img_feat_3crop_norm1.h5'
            #feat_file = './train_txt/train_cont.txt'
            #feat_file = './news_validate_end_info.npy'
            #feat_file = './fusai_news_end_info.npy'
            #feat_file = './fusai_news_info.npy'
            #feat_file = './val_img_feat_3crop_norm1.h5'
            #feat_file = './clear_news_info.npy'
            #feat_file ='./news_end_info.npy'
            model_path = './model/Bidirectionnet_lstm/model-9000'
            #scope = 'sentence'
            scope='image'
            save_path ='./emb/train_image_embed_lstm_9000.h5'
            #save_path = './emb/train_sentence_embed_lstm_9000.h5'
            model.test_embed(sess, feat_file, model_path, scope, save_path)	    







