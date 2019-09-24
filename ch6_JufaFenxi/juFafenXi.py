# -*- coding:utf8 -*-

# 分词
import jieba
string = '他骑自行车去了菜市场'
seg_list = jieba.cut(string, cut_all=False, HMM=True)
seg_str = ' '.join(seg_list)
print(seg_str)

# PCFG句法分析
import os

root = './'
parser_path = root + 'stanford-parser.jar'
model_path = root + 'stanford-parser-3.8.0-models.jar'

# 指定jdk路径
if not os.environ.get('JAVA_HOME'):
    JAVA_HOME = 'C:\Program Files\Java\jdk1.8.0_151'
    os.environ['JAVA_HOME'] = JAVA_HOME


from nltk.tree import Tree
from stanfordcorenlp import StanfordCoreNLP

with StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05', lang = 'zh') as nlp:
    sentence = nlp.parse(seg_str)
    print(sentence)
    t = Tree.fromstring(sentence)
    print(t)
    print(t.__repr__())
    print(t.draw())

    print("----------Display tree properties:----------")
    print(t.label())  # tree's constituent type
    print(t.height())
    print(t.leaves())
