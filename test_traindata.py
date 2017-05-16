import h5py
import numpy as np
import tensorflow as tf
x=h5py.File('./emb/train_sentence_embed_lstm_9000.h5','r')
s_f = np.array(x['embed'])
#x=h5py.File('./emb/train_end_sentence_embed_pca_simpleconv_crop3_fewwords_ftlm11500.h5','r')
#s_end_f = np.array(x['embed'])
x=h5py.File('./emb/train_image_embed_lstm_9000.h5','r')
image_f = np.array(x['embed'])
#x=h5py.File('./train_img_feat_3crop_pca512.h5','r')
#image_f = np.array(x['feature'])

s = s_f[:400].dot(image_f.T)
r= np.array([sum(s[v]>s.diagonal()[v]) for v in range(400)])
#s_total = s_start+s_end
#r_total= np.array([sum(s_total[v]>s_total.diagonal()[v]) for v in range(400)])