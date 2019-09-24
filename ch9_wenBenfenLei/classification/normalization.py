# -*- coding: utf-8 -*-

"""
@Time    : 2019/9/5 10:16
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
with open("dic/stop_words.utf8",encoding="utf-8") as f:
    stopword_list = f.readlines()


def tokenize_text(text):
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens

def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('',token) for token in tokens])
    filtered_text = ''.join(filtered_tokens)
    return filtered_text

def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text

def normalize_corpus(corpus, tokenize= False):
    normalize_corpus = []
    for text in corpus:
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalize_corpus.append(text)
        if tokenize:
            text = tokenize_text(text)
            normalize_corpus.append(text)
    return normalize_corpus














"""
# 去除字符串内标点的几种方法的效率对比
import re, string, timeit

s = "string. With. Punctuation"
exclude = set(string.punctuation)
table = string.maketrans("","")
regex = re.compile('[%s]' % re.escape(string.punctuation))

def test_set(s):
    return ''.join(ch for ch in s if ch not in exclude)

def test_re(s):  # From Vinko's solution, with fix.
    return regex.sub('', s)

def test_trans(s):
    return s.translate(table, string.punctuation)

def test_repl(s):  # From S.Lott's solution
    for c in string.punctuation:
        s=s.replace(c,"")
    return s

print ("sets      :",timeit.Timer('f(s)', 'from __main__ import s,test_set as f').timeit(1000000))
print ("regex     :",timeit.Timer('f(s)', 'from __main__ import s,test_re as f').timeit(1000000))
print ("translate :",timeit.Timer('f(s)', 'from __main__ import s,test_trans as f').timeit(1000000))
print ("replace   :",timeit.Timer('f(s)', 'from __main__ import s,test_repl as f').timeit(1000000))

"""
