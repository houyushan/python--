# -*- coding: utf-8 -*-

"""
@Time    : 2019/9/7 9:55
@User    : HouYushan
@Author  : xueba1521
@FileName: test.py
@Software: PyCharm
@Blog    ：http://---
"""

import pandas as pd
from normalization import *
# 加载停用词
with open("dict/stop_words.txt", encoding="utf8") as f:
    stopword_list = f.readlines()
print(stopword_list)

book_data = pd.read_csv('data/data.csv', encoding='utf8') # 读取文件
print(book_data.head())
book_titles = book_data['title'].tolist()
book_content = book_data['content']
book_content = book_content.astype(str).tolist()
print('书名:', book_titles[0])
print('内容:', book_content[0][:10])

def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text

import string ,re

sss = "现代人内心流失的东西，,,,，，，，，,,,,,,,，，，，，，，，这家杂货店能帮你找回—— 。。。。。"
print('0', sss)
regex = re.compile('[%s]' % re.escape(string.punctuation))
def test_re(s):
    return regex.sub('', s)
ts = test_re(sss)
print('1',ts)

def t(n):
    return n.split('\n')[0]

for i in ts:
    ll = list(map(t, stopword_list))
    if i not in ll:
        print(i)
    # for j in stopword_list:
    #     if i != j:
    #         print(type(i), type(j))
    #         print('i:', i, 'j:', j)
# print('2',tss)


# book_content1 = []
# for book_one in book_content:
#     print(type(book_one))
#     print(book_one)
#     book_text = tokenize_text(book_one)
#     print(type(book_text))
#     l = []
#     for s in book_text:
#         ss = test_re(s)
#         l.append(ss)
#     book_content1.append(l)
#     break
# print(book_content1)

# i: 。 j: 。

lisss = ['。', '?', ',', '，']
new_j = []
def test(str1, str2):
    for j in stopword_list:
        j1 = j.split('\n')[0]
        new_j.append(j1)
    print(new_j)
    def t(n):
        return n.split('\n')[0]
    ll = list(map(t, stopword_list))
    if str1 in ll:
        print(True)

test('。', '。')