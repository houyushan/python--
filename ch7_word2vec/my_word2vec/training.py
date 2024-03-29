# -*- coding: utf8 -*-

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
from numba import jit

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
@jit
def my_function():
    wiki_news = open('./data/reduce_zhwiki1.txt', 'r', encoding='utf8')
    model = Word2Vec(LineSentence(wiki_news), sg=0, size=192, window=5, min_count=5, workers=9)
    model.save('./data/zhiwiki_news.word2vec')

if __name__ == '__main__':
    my_function()