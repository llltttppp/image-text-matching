import tensorflow as tf
import numpy as np
import h5py
import time
import sys
slim = tf.contrib.slim
class BidirectionNet:
    def __init__(self,is_training=True):
        self.weight_decay = 0.0001
        self.endpoint={}
        self.is_training = is_training
        self.keep_prob = 0.5 if is_training else 1.0
        self.build_input()
        self.build_matchnet()
        if is_training:
            self.build_summary()
    def build_input(self):
        # positive
        self.labels = tf.placeholder(tf.float32, shape=[None,512], name='concept_labels')
        self.image_feat = tf.placeholder(tf.float32,shape=[None,3,4096], name='image_features')   
    def conv_layer(self, X, num_output, kernel_size, s, p='SAME'):
        return tf.contrib.layers.conv2d(X,num_output,kernel_size,s,\
                                        padding=p,weights_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),\
                                        normalizer_fn=tf.contrib.layers.batch_norm,normalizer_params={'is_training':self.is_training,'updates_collections':None})
    def sentencenet(self, sentence_emb, reuse=False):
        with tf.variable_scope('sentence_net', reuse=reuse) as scope:
            conv = self.conv_layer
            wd = tf.contrib.layers.l2_regularizer(self.weight_decay)
            conv1 = conv(sentence_emb,512,(1,5),1)
            conv1 = tf.nn.max_pool(conv1,[1,1,2,1],[1,1,2,1],padding='SAME')
            conv2 = conv(conv1,512,[1,3],1)
            conv2 = conv(conv2,512,[1,3],1)
            conv2 = tf.nn.max_pool(conv2,[1,1,2,1],[1,1,2,1],padding='SAME')
            conv3 = conv(conv2,256,[1,3],1)
            conv3 = conv(conv3,256,[1,3],1)
            conv3 = tf.nn.max_pool(conv3,[1,1,2,1],[1,1,2,1],padding='SAME')
            conv4 = conv(conv3,256,[1,2],1)
            conv4 = tf.nn.max_pool(conv4,[1,1,2,1],[1,1,2,1],padding='SAME')             
            conv4 = conv(conv4,128,[1,2],1)
            conv4 = tf.nn.max_pool(conv4,[1,1,2,1],[1,1,2,1],padding='SAME')   
            s = tf.shape(self.raw_sentence)
            sentence_fc1 =tf.nn.dropout(tf.contrib.layers.fully_connected(tf.reshape(conv4,[s[0], 25 * 128]),2048, \
                                                                          weights_regularizer=wd, scope='s_fc1'), self.keep_prob) # 25 * 128
            sentence_fc2 = tf.contrib.layers.fully_connected(sentence_fc1, 512,activation_fn=None,normalizer_fn=tf.contrib.layers.batch_norm,\
                                                             normalizer_params={'is_training':self.is_training,'updates_collections':None}, weights_regularizer=wd, scope='s_fc2')
            sentence_fc2 = sentence_fc2/tf.norm(sentence_fc2,axis= -1,keep_dims=True)
        self.endpoint['sentence_conv4'] = conv4
        self.endpoint['sentence_fc2'] = sentence_fc2
        return sentence_fc2
    def imagenet(self, image_feat, reuse=False):
        with tf.variable_scope('image_net', reuse=reuse) as scope:
            wd = tf.contrib.layers.l2_regularizer(self.weight_decay)
            image_fc1 = tf.contrib.layers.fully_connected(image_feat,2048, weights_regularizer=wd,scope='i_fc1')
            image_fc2 = tf.contrib.layers.fully_connected(image_fc1, 512, activation_fn=tf.nn.sigmoid, normalizer_fn=tf.contrib.layers.batch_norm, \
                                                         normalizer_params={'is_training':self.is_training,'updates_collections':None}, weights_regularizer=wd, scope='i_fc2')
            prob = 1-tf.reduce_prod(1-image_fc2,axis=1)
        self.endpoint['image_fc1'] = image_fc1
        self.endpoint['image_fc2'] = image_fc2
        self.endpoint['prob'] = prob
        return prob
    def mil_loss(self,prob,labels):        
        cross_entropy = -tf.reduce_sum(labels*tf.log(prob) + (1-labels)*tf.log(1-prob))
        return cross_entropy
    def triplet_loss(self, common, pos, neg, margin=0.2):
        # d(common, pos) + margin < d(common, neg)
        self.d_pos = tf.reduce_sum(tf.squared_difference(common, pos),-1)
        self.d_neg =tf.reduce_sum(tf.squared_difference(common, neg),-1)
        return tf.reduce_sum(tf.nn.relu(self.d_pos + margin - self.d_neg, name = 'triplet_loss'))
    
    def build_matchnet(self):
        self.prob = self.imagenet(self.image_feat, reuse=False,skip=self.is_skip)
        # compute loss
        if self.is_training:
            
            self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.total_loss = tf.add_n([self.mil_loss(self.prob,self.labels)]+self.reg_loss)
            self.saver = tf.train.Saver(max_to_keep=20)
    def build_summary(self):
        tf.summary.scalar('loss/reg_loss', tf.add_n(self.reg_loss))
        tf.summary.scalar('loss/total_loss', self.total_loss)      
        for name, tensor in self.endpoint.items():
            tf.summary.histogram('activation/' + name, tensor)
        
        t_var = tf.trainable_variables()
        watch_list = ['i_fc1', 'i_fc2'] 
        for watch_scope in watch_list:
            watch_var = [var for var in t_var if watch_scope+'/weights' in var.name]
            tf.summary.histogram('weights/'+watch_scope, watch_var[0])
    def positive_loss(self, x, y):
        return tf.reduce_sum(tf.squared_difference(x, y, name='positive_loss'))        
    def top_K_loss(self,sentence,image,K=30,margin=0.3):
        sim_matrix = tf.matmul(sentence, image,transpose_b=True)
        bs = tf.shape(sim_matrix)[0]
        s_square = tf.reduce_sum(tf.square(sentence),axis=1)
        im_square =tf.reduce_sum(tf.square(image),axis=1)
        d = tf.reshape(s_square,[-1,1])-2*sim_matrix+tf.reshape(im_square,[1,-1])
        positive = tf.stack([tf.matrix_diag_part(d)]*K,1)
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
        return tf.reduce_sum(sentence_center_loss),tf.reduce_sum(image_center_loss)
        
        
        
    def train(self, sess, maxEpoch=30, batch_size=100, lr=0.001,is_load=False, ckpt_path=''):
        logdir = './log/run_fixnetpcaTopK'
        model_save_path='./model/fixnetpcaTopK/model'
        sentence = np.load('./clear_news_info.npy')
        h5file = h5py.File('./train_img_feat_pca512.h5', mode='r')
        image_feat_all = np.array(h5file['feature'])
        opt =  tf.train.AdamOptimizer(learning_rate=lr)
        train_op1 =opt.minimize(self.total_loss,var_list=[v for v in tf.trainable_variables() if 'image_net' not in v.name])
        train_op2 =opt.minimize(self.total_loss)
                                    
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()
        N = sentence.shape[0]
        assert N == image_feat_all.shape[0]
        batch_idx = int(N / batch_size)
        sess.run(tf.global_variables_initializer())
        if is_load:
            self.saver.restore(sess, ckpt_path)
            print '%s loaded' %ckpt_path        
        step = 4000
        t0 = time.time()
        for epoch in range(maxEpoch):
            if epoch  < maxEpoch*0.0:
                train_op = train_op1
            else:
                train_op = train_op2
            # shuffle
            idxArr = np.random.permutation(N)
            for idx in range(batch_idx):
                interval = range(idx*batch_size , (idx+1)*batch_size)
                raw_sentence = sentence[idxArr[interval]]
                image_feat = image_feat_all[idxArr[interval]]
                # sample
                # train
                feed_dict = {self.raw_sentence: raw_sentence, self.image_feat: image_feat}
                _, summary, total_loss = sess.run([train_op, summary_op, self.total_loss], feed_dict=feed_dict)
                
                if np.mod(step, 10) == 0:
                    summary_writer.add_summary(summary, global_step=step)
                if np.mod(step, 1000) == 0:
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
        load_var = [var for var in t_var if scope in var.name]
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

if __name__ == '__main__':
    is_train = False
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model = BidirectionNet(is_training=is_train,is_skip=True)
        if is_train:
            model.train(sess,lr=0.00001)
            #model.train(sess,lr=0.0001,is_load=True,ckpt_path='./model/fixnetpcaTopK/model-4000')
        else:
            #feat_file = './train_img_feat.h5'
            #feat_file = './news_validate_info_sub1.npy'
            feat_file = './clear_news_info.npy'
            model_path = './model/fixnetpcaTopK/model-10000'
            scope = 'sentence'
            #scope='image'
            #save_path = './emb/train_image_embed_11k.h5'
            save_path = './emb/train_sentence_embed_pcaTopK1w.h5'
            model.test_embed(sess, feat_file, model_path, scope, save_path,batch_size=149)
    
            tf.random_crop
            
            
            
            
            
