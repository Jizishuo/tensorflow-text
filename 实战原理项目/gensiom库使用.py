from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s', level=logging.INFO)

raw_sentences = ["the quick brown fox jumpa over the lazy dogs", "lalala you go home now to sleep"]

sentences = [s.split() for s in raw_sentences]
print(sentences)

#min_count 把出现频率低的词过滤掉(0-100)
#size 神经网络的层数 默认100， 设置为100的倍数

models = word2vec.Word2Vec(sentences, min_count=1)
print(models.similarity('dogs', 'you')) #对比2个词的相识度