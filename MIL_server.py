# -*- coding:utf-8 -*- 
import tensorflow as tf
import numpy as np
import h5py
import time
import sys
import os
slim = tf.contrib.slim
def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print 'folder %s created' %path
class BidirectionNet:
    def __init__(self,is_training=True,is_keep_prob=False):
        self.weight_decay = 0.00005
        self.num_class=1000
        self.instance_list ={}
        self.instance_list.update([(v.strip().split(' ')[0],int(v.strip().split(' ')[1])) for v in open('./vocabulary/keywords_labels.txt','r').readlines()])                               
        self.cluster_idf = np.load('cluster_idf.npy')
        self.endpoint={}
        self.is_training = is_training
        self.keep_prob = 0.5 if is_keep_prob else 1.0
        self.build_input()
        self.build_matchnet()
        if is_training:
            self.build_summary()
    def build_input(self):
        # positive
        self.labels = tf.placeholder(tf.float32, shape=[None,self.num_class], name='concept_labels')
        self.image_feat = tf.placeholder(tf.float32,shape=[None,3,4096], name='image_features')   
    def conv_layer(self, X, num_output, kernel_size, s, p='SAME'):
        return tf.contrib.layers.conv2d(X,num_output,kernel_size,s,\
                                        padding=p,weights_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),\
                                        normalizer_fn=tf.contrib.layers.batch_norm,normalizer_params={'is_training':self.is_training,'updates_collections':None})
    
    def imagenet(self, image_feat, reuse=False):
        with tf.variable_scope('image_net', reuse=reuse) as scope:
            wd = tf.contrib.layers.l2_regularizer(self.weight_decay)
            image_fc1 = tf.contrib.layers.fully_connected(image_feat,4096, weights_regularizer=wd,scope='i_fc1')
            image_fc2 = tf.contrib.layers.fully_connected(image_fc1, self.num_class, activation_fn=None,weights_regularizer=wd, scope='i_fc2')
            prob = tf.reduce_mean(image_fc2,axis=1)#1-tf.reduce_prod(1-image_fc2,axis=1)
        self.endpoint['image_fc1'] = image_fc1
        self.endpoint['image_fc2'] = image_fc2
        self.endpoint['prob'] = prob
        return prob
    def softmax_loss(self,image_fc,labels):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, 
                                                      logits=image_fc))
    def mil_loss(self,prob,labels,elips=1e-12,postive_weight=1.0):  
        prob1 =prob
        self.ploss=-tf.reduce_sum(labels*tf.log(prob1+elips))#/tf.reduce_sum(labels)
        topk_neg=tf.nn.top_k(-(1-labels)*tf.log(1-prob1+elips),k=10)[0]
        self.nloss=tf.reduce_sum(topk_neg)
        cross_entropy = postive_weight*self.ploss+self.nloss
        return cross_entropy
    def triplet_loss(self, common, pos, neg, margin=0.2):
        # d(common, pos) + margin < d(common, neg)
        self.d_pos = tf.reduce_sum(tf.squared_difference(common, pos),-1)
        self.d_neg =tf.reduce_sum(tf.squared_difference(common, neg),-1)
        return tf.reduce_sum(tf.nn.relu(self.d_pos + margin - self.d_neg, name = 'triplet_loss'))
    
    def build_matchnet(self):
        self.prob = self.imagenet(self.image_feat, reuse=False)
        # compute loss
        if self.is_training:           
            self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.total_loss = tf.add_n([self.softmax_loss(self.prob,self.labels)]+self.reg_loss)
            #self.total_loss = tf.add_n([self.mil_loss(self.prob,self.labels)]+self.reg_loss)
            self.saver = tf.train.Saver(max_to_keep=50)
    def build_summary(self):
        tf.summary.scalar('loss/reg_loss', tf.add_n(self.reg_loss))
        tf.summary.scalar('loss/total_loss', self.total_loss) 
        #tf.summary.scalar('loss/ploss', self.ploss)
        #tf.summary.scalar('loss/nloss', self.nloss)        
        for name, tensor in self.endpoint.items():
            tf.summary.histogram('activation/' + name, tensor)
        
        t_var = tf.trainable_variables()
        watch_list = ['i_fc1', 'i_fc2'] 
        for watch_scope in watch_list:
            watch_var = [var for var in t_var if watch_scope+'/weights' in var.name]
            try:
                tf.summary.histogram('weights/'+watch_scope, watch_var[0])
            except:
                pass
    def build_trainop(self,loss,lr=0.001,clipping_norm=100,optimizer =tf.train.AdamOptimizer,tvars=None):
        if tvars is None:        
            tvars = tf.trainable_variables()
        g=tf.gradients(loss, tvars)
        grads= [tf.clip_by_norm(v,clipping_norm)  for v in g ]
        opt = optimizer(lr)
        for i,v in enumerate(tvars):
            tf.summary.histogram(name=v.name+'_gradient', values=grads[i])
        return opt.apply_gradients(zip(grads,tvars))    
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
        
        
        
    def train(self, sess, maxEpoch=50, batch_size=2000, lr=0.001,is_load=False, ckpt_path=''):
        logdir = './log/mil/keyword'
        print 'log in %s' %logdir
        model_save_path='./model/mil/keyword/'
        make_if_not_exist(model_save_path)
        model_save_path += 'model'        
    
        data_root = '/media/ltp/40BC89ECBC89DD32/souhu_fusai/'
        sentence_all = np.array([v.strip() for v in open('./train_txt/train.txt').readlines()])
        
        img_feat_file = data_root + 'train_img_feat_3crop_notAvg_all.h5'
        print 'image feature read from %s' %img_feat_file	     
        img_h5 = h5py.File(img_feat_file, 'r')
        L_im = img_h5['feature'].shape[0]
        nDataset = 6
        dataset_size = int(L_im/nDataset)        
        
        step = 0
        train_op=self.build_trainop(self.total_loss,lr=lr)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())        
                        
        if is_load:
            self.saver.restore(sess, ckpt_path)
            print '%s loaded' %ckpt_path        
        step = 0
        t0 = time.time()
        
        for epoch in range(maxEpoch):
            for setId in np.random.permutation(nDataset):
                shift = setId * dataset_size
                if setId < nDataset - 1:
                    img_feat_all = img_h5['feature'][shift:shift+dataset_size,:]
                    sentence = sentence_all[shift:shift+dataset_size]
                    N = img_feat_all.shape[0]                     
                    assert N == dataset_size
                else: 
                    img_feat_all = img_h5['feature'][shift:,:]
                    sentence = sentence_all[shift:]
                    N = img_feat_all.shape[0]   
                print 'load No %d dataset'%setId
                for _ in range(1):
                    idxArr = np.random.permutation(N)
                    batch_idx = int(N / batch_size)
                    for idx in range(batch_idx):
                        interval = range(idx*batch_size , (idx+1)*batch_size)
                        raw_sentence = self.doc2vector(sentence[idxArr[interval]],batch_size)
                        image_feat = img_feat_all[idxArr[interval]]
                        #lda_feat = lda[idxArr[interval]]
        
                        # train
                        #feed_dict = {self.tfidf_feat: tfidf_feat, self.lda_feat: lda_feat, self.image_feat: image_feat, self.lr: lr}
                        feed_dict = {self.labels:raw_sentence, self.image_feat: image_feat}
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
    def doc2vector(self,doc_list,batchsize = 100,max_words=1):
        #ss=time.time()
        labels =np.zeros((batchsize,self.num_class),dtype=np.float32)
        count=0
        for i,v in enumerate(doc_list):
            tmp_labels =np.zeros((self.num_class,),dtype = np.float32)
            for word in v.split(' '):
                try:
                    tmp_labels[self.instance_list[word]]+=1*self.cluster_idf[self.instance_list[word]]
                    count+=1
                except:
                    pass
            labels[i,np.argsort(-tmp_labels)[:max_words]]=1
       # print time.time()-ss
        return labels
            
    def test_embed(self, sess, feat_file, model_path, scope, save_path, h5dataset='embed', batch_size=100):
        '''
        For testing. Generate the final embedding of images for image-text matching.
        Dataset: 'embed'
        Scope: either 'image' or 'sentence'
        ''' 
        # read input features
        if scope == 'image':
            target_tensor = self.endpoint['prob']
            input_tensor = self.image_feat
            h5file = h5py.File(feat_file, 'r')
            feat_all = np.array(h5file['feature'])
            h5file.close()            
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
        model = BidirectionNet(is_training=is_train)
        if is_train:
            #model.train(sess,lr=0.00,batch01,batch_size=2000)
            model.train(sess,lr=0.0001,
                        batch_size=1000)#,is_load=True,ckpt_path='./model/mil/model-78000')
        else:
            #feat_file = './train_img_feat.h5'
            #feat_file = './news_validate_info_sub1.npy'
            feat_file = './test_img_feat_3crop_notAvg.h5'
            model_path = './model/mil/keyword/model-60000'
            #scope = 'sentence'
            scope='image'
            save_path = './emb/train_image_embed_mil_softmax_60000.h5'
            #save_path = './emb/train_sentence_embed_pcaTopK1w.h5'
            model.test_embed(sess, feat_file, model_path, scope, save_path,batch_size=1000)
            text = np.array([v.strip() for v in open('./test_txt/test.txt').readlines()])
            np.save('./emb/train_sentence_labels.npy',model.doc2vector(doc_list=text, batchsize=len(text), max_words=3))
    
            
            
            
            
            
