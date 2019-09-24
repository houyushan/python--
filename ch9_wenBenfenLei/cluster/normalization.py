# -*- coding: utf-8 -*-

"""
@Time    : 2019/9/6 13:58
@User    : HouYushan
@Author  : xueba1521
@FileName: normalization.py
@Software: PyCharm
@Blog    ：http://---
"""
import re
import string
import jieba

# 加载停用词
with open("dict/stop_words.utf8", encoding="utf8") as f:
    stopword_list = f.readlines()
def t(n):
    return n.split('\n')[0]
stopword_list = list(map(t, stopword_list))


def tokenize_text(text):
    tokens = jieba.lcut(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus):
    normalized_corpus = []
    for text in corpus:

        text =" ".join(jieba.lcut(text))
        normalized_corpus.append(text)

    return normalized_corpus
