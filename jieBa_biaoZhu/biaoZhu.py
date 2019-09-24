# -*- coding=utf8 -*-

import re
re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")

import jieba.posseg as psg

sent = '中文分词是文本处理不可或缺的一步!'
seg_list = psg.cut(sent)
print(' '.join(['{0}/{1}'.format(w,t) for w,t in seg_list]))

