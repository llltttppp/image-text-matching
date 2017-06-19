from gensim.models.wrappers import FastText
model=FastText()
model.load_word2vec_format('/home/ltp/WorkShop/fastText/model/wiki.zh.vec')