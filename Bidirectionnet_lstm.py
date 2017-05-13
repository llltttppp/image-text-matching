import tensorflow as tf
import numpy as np
import h5py
import time
import sys
slim = tf.contrib.slim
class BidirectionNet:
    def __init__(self,is_training=True,is_skip=False,is_TopKloss=True,word2vec_model='/media/ltp/625C80425C8012C9/word2vec/cn.cbow.bin'):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True, unicode_errors='ignore')
        
        self.weight_decay = 0.0001
        self.endpoint={}
        self.is_skip=is_skip
        self.is_TopKloss = is_TopKloss
        self.is_training = is_training
        self.keep_prob = 0.5 if is_training else 1.0
        self.build_input()
        self.build_matchnet()
        if is_training:
            self.build_summary()
    def build_input(self):
        # positive
        self.vector_size=300
        self.sentence_len =100
        self.raw_sentence= tf.placeholder(tf.int32, shape=[None,self.sentence_len,self.vector_size],name='raw_sentence')
        self.sentence_emb =self.raw_sentence #tf.nn.embedding_lookup(tf.get_variable('word_embedding',[4096,512]),self.raw_sentence)
        if self.is_skip:
            self.image_feat = tf.placeholder(tf.float32,shape=[None,512], name='image_features')
        else:
            self.image_feat = tf.placeholder(tf.float32,shape=[None,4096], name='image_features')   
    def conv_layer(self, X, num_output, kernel_size, s, p='SAME'):
        return tf.contrib.layers.conv2d(X,num_output,kernel_size,s,\
                                        padding=p,weights_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),\
                                        normalizer_fn=tf.contrib.layers.batch_norm,normalizer_params={'is_training':self.is_training,'updates_collections':None})
    def sentencenet(self, sentence_emb, reuse=False):
        with tf.variable_scope('sentence_net', reuse=reuse) as scope:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=300)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,input_keep_prob=self.keep_prob,output_keep_prob=self.keep_prob)
            zero_state = lstm_cell.zero_state(
                batch_size=self.sentence_emb.get_shape()[0], dtype=tf.float32)
                        
            input_list = tf.unstack(self.sentence_emb,axis=1)
            output,_ = tf.contrib.rnn.static_rnn(lstm_cell, inputs=input_list,initial_state=zero_state)
            lstm_output = output[-1]
            sentence_fc1 =tf.contrib.layers.fully_connected(lstm_output,2048, \
                                                            weights_regularizer=wd, scope='s_fc1') # 20*10*256
            sentence_fc2 = tf.contrib.layers.fully_connected(sentence_fc1, 512,activation_fn=None,normalizer_fn=tf.contrib.layers.batch_norm,\
                                                             normalizer_params={'is_training':self.is_training,'updates_collections':None}, weights_regularizer=wd, scope='s_fc2')
            sentence_fc2 = sentence_fc2/tf.norm(sentence_fc2,axis= -1,keep_dims=True)
        self.endpoint['sentence_lstm'] = lstm_output
        self.endpoint['sentence_fc2'] = sentence_fc2
        return sentence_fc2
    def imagenet(self, image_feat, reuse=False,skip=False):
        if skip:
            return image_feat
        with tf.variable_scope('image_net', reuse=reuse) as scope:
            wd = tf.contrib.layers.l2_regularizer(self.weight_decay)
            image_fc1 = tf.contrib.layers.fully_connected(image_feat,2048, weights_regularizer=wd,scope='i_fc1')
            image_fc2 = tf.contrib.layers.fully_connected(image_fc1, 512, activation_fn=None, normalizer_fn=tf.contrib.layers.batch_norm, \
                                                          normalizer_params={'is_training':self.is_training,'updates_collections':None}, weights_regularizer=wd, scope='i_fc2')
            image_fc2 = image_fc2 / tf.norm(image_fc2,axis=-1,keep_dims=True)
        self.endpoint['image_fc1'] = image_fc1
        self.endpoint['image_fc2'] = image_fc2
        return image_fc2
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
        image =tf.stack([image]*20,axis=1)
        diff = tf.reduce_sum(tf.squared_difference(sentence, image, name='positive_loss'),axis=-1) 
        diff = tf.reduce_min(diff,axis=-1)
        return tf.reduce_sum(diff)       
    def top_K_loss(self,sentence_stack,image,K=50,margin=0.4):
        imagestack =tf.stack([image]*20,axis=1)
        ishape = tf.shape(imagestack)[0]
        slice_id = tf.cast(tf.argmin(tf.reduce_sum(tf.squared_difference(sentence_stack, imagestack),axis=-1),axis=1),tf.int32)
        sentence = tf.gather_nd(sentence_stack,indices=tf.concat([tf.expand_dims(tf.range(ishape),1),tf.expand_dims(slice_id,1)],1))
        sentence_T=tf.transpose(sentence_stack,[1,0,2])
        image_T=tf.transpose(imagestack,[1,2,0])
        sim_matrix = tf.reduce_max(tf.matmul(sentence_T, image_T),0)
        bs = tf.shape(sim_matrix)[0]
        s_square = tf.reduce_sum(tf.square(sentence),axis=-1)
        im_square =tf.reduce_sum(tf.square(image),axis=-1)

        d = tf.reshape(s_square,[-1,1])-2*sim_matrix+tf.reshape(im_square,[1,-1])
        positive = tf.stack([tf.matrix_diag_part(d)]*K,1)
        positive_cos = tf.reduce_sum(tf.squared_difference(sentence,image),axis=-1)
        length = tf.shape(d)[-1]
        d = tf.matrix_set_diag(d, 100*tf.ones([length]))
        sen_loss_K ,_= tf.nn.top_k(-d,K,sorted=False)
        im_loss_K,_=tf.nn.top_k(tf.transpose(-d),K,sorted=False)
        sentence_center_loss = tf.nn.relu(sen_loss_K + positive +margin)
        image_center_loss = tf.nn.relu(im_loss_K + positive +margin)
        self.d_neg =tf.reduce_mean(-sen_loss_K-im_loss_K)/2
        self.d_pos = tf.reduce_mean(positive)
        self.endpoint['debug/sentence_center_loss']=sentence_center_loss
        self.endpoint['debug/image_center_loss']=image_center_loss
        self.endpoint['debug/sim_matrix']=sim_matrix
        self.endpoint['debug/sen_loss_K']=-sen_loss_K
        self.endpoint['debug/image_loss_K']=-im_loss_K
        self.endpoint['debug/distance']=d
        self.endpoint['debug/positive']=positive
        self.endpoint['debug/positive_cos']=positive_cos
        return tf.reduce_sum(sentence_center_loss),tf.reduce_sum(image_center_loss)



    def train(self, sess, maxEpoch=300, batch_size=500, lr=0.0001,is_load=False,only_sentence=False,is_fixsentence=False, ckpt_path=''):
        logdir = './log/run_Bidirectionnet_newconv_skip'
        model_save_path='./model/Bidirectionnet_newconv_skip/model'
        sentence = np.array(open('train.txt','r').readlines())
        h5file = h5py.File('./train_img_feat_3crop_pca512.h5', mode='r')
        image_feat_all = np.array(h5file['feature'])
        opt =  tf.train.AdamOptimizer(learning_rate=lr)
        train_op1 =opt.minimize(self.total_loss,var_list=[v for v in tf.trainable_variables() if ('word_embedding' not in v.name) and ('sentence_net' not in v.name)]+[v for v in tf.trainable_variables() if 's_fc' in v.name])
        train_op2 =opt.minimize(self.total_loss)

        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()
        N = image_feat_all.shape[0]
        batch_idx = int(N / batch_size)
        sess.run(tf.global_variables_initializer())
        if is_load:
            if only_sentence:
                ssaver = tf.train.Saver([var for var in tf.trainable_variables() if (('s_fc' not in var.name) and ('i_fc' not in var.name))])
                ssaver.restore(sess, ckpt_path)
                print '%s loaded' %ckpt_path   
            else:
                self.saver.restore(sess, ckpt_path)
                print '%s loaded' %ckpt_path       


        step =0
        t0 = time.time()
        for epoch in range(maxEpoch):
            if is_fixsentence:
                train_op = train_op1
            else:
                train_op = train_op2
            # shuffle
            assert N==sentence.shape[0]
            idxArr = np.random.permutation(N)
            for idx in range(batch_idx):
                interval = range(idx*batch_size , (idx+1)*batch_size)
                raw_sentence = self.read_wordvector(sentence[idxArr[interval]],batch_size)
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
        elif scope == 'sentence':
            target_tensor = self.sentence_fc2
            input_tensor = self.raw_sentence
            feat_all = np.load(feat_file)
        else:
            print 'invalid scope %s (must be either image or sentence)' %target_tensor.name
            sys.exit(1)   
        #feat_all = feat_all[:10000, ...]
        N = feat_all.shape[0]
        assert np.mod(N, batch_size) == 0
        # load model
        t_var = tf.global_variables()
        load_var = [var for var in t_var ]
        loader = tf.train.Saver(var_list = load_var)
        loader.restore(sess, model_path)
        # forward
        embed = []
        t0 = time.time()
        for idx in range(N/batch_size):
            interval = np.array(range(idx * batch_size, (idx + 1) * batch_size))
            feat_batch = feat_all[interval]
            feed_dict = {input_tensor: feat_batch}
            embed_batch = sess.run(target_tensor, feed_dict=feed_dict)
            embed.extend(embed_batch)
            if np.mod(idx, 10) == 0:
                t = (time.time() - t0)/60
                print '%.2f minutes. Iteration %d/%d' %(t, idx, N/batch_size)
        # save embed
        embed = np.array(embed)
        h5file = h5py.File(save_path, 'w')
        h5file.create_dataset(h5dataset, data=embed)
        h5file.close()
        print 'embed done for scope %s. Saved shape ' %scope, embed.shape


    def read_wordvector(self,batch_sentences,batch_size):
        batch_vectors=np.zeros([batch_size,self.sentence_len,self.vector_size])
        for i,v in enumerate(batch_sentences):
            sentence_matrix = np.zeros([self.sentence_len,self.vector_size])
            for j,word in enumerate(v.split(' ')):
                wordvec = self.model.word_vec(word)
                sentence_matrix[j,:]=wordvec
            batch_vectors[i,:,:]=sentence_matrix
        return batch_vectors
        

    
if __name__ == '__main__':
    is_train = False
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model = BidirectionNet(is_training=True,is_skip=True,is_TopKloss =True)
        if is_train:
            model.train(sess,lr=0.0001)
            #model.train(sess,lr=0.0001,only_sentence=False,is_load=True,is_fixsentence=True,ckpt_path='./model/fixnetpcaTopK_simpleconv_crop3_lm/model-6500')
        else:
            #feat_file = './train_img_feat_3crop_norm1.h5'
            feat_file = './news_validate_info_sub1.npy'
            #feat_file = './news_validate_end_info.npy'
            #feat_file = './fusai_news_end_info.npy'
            #feat_file = './fusai_news_info.npy'
            #feat_file = './val_img_feat_3crop_norm1.h5'
            #feat_file = './clear_news_info.npy'
            #feat_file ='./news_end_info.npy'
            model_path = './model/Bidirectionnet_newconv_skip/model-55500'
            scope = 'sentence'
            #scope='image'
            #save_path ='./emb/train_image_embed_newconv_5000.h5'
            save_path = './emb/validate_sentence_embed_newconv_skip_55500.h5'
            model.test_embed(sess, feat_file, model_path, scope, save_path,batch_size=100)	    







