# -*- coding: utf8 -*-

import numpy as np
import time


wordsList = np.load('./wordsList.npy')
print('载入word列表')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]


wordVectors = np.load('./wordVectors.npy')
print('载入文本向量')

print(len(wordsList))
print(wordVectors.shape)

start_t = time.time()
import os
from os.path import isfile, join
pos_files = ['pos/' + f for f in os.listdir('pos/') if isfile(join('pos/', f))]
neg_files = ['neg/' + f for f in os.listdir('neg/') if isfile(join('neg/', f))]

num_words = []
for pf in pos_files:
    with open(pf, 'r', encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('正面评价完结')

for nf in neg_files:
    with open(nf, 'r', encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)

print('负面评价结束')

num_files = len(num_words)
print('文件总数：', num_files)
print('所有的词的数量：', sum(num_words))
print('平均文件词的长度：', sum(num_words)/len(num_words))

end_t = time.time()
print(end_t - start_t)

# 进行可视化
import matplotlib
matplotlib.use('qt4agg')
# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
# %matplotlib inline
import matplotlib.pyplot as plt
plt.hist(num_words, 50, facecolor='g')
plt.xlabel('文本长度')
plt.ylabel('频次')
plt.axis([0, 1200, 0, 8000])
plt.show()

# 将文本生成一个索引矩阵
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
num_dimensions = 300 # 指定每个文本的维度

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


max_seq_num = 300
# ids = np.zeros((num_files, max_seq_num), dtype='int32')
# file_count = 0
# for pf in pos_files:
#     with open(pf, "r", encoding='utf-8') as f:
#         indexCounter = 0
#         line = f.readline()
#         cleanedLine = cleanSentences(line)
#         split = cleanedLine.split()
#         for word in split:
#             try:
#                 ids[file_count][indexCounter] = wordsList.index(word)
#             except ValueError:
#                 ids[file_count][indexCounter] = 399999
#             indexCounter += 1
#             if indexCounter >= max_seq_num:
#                 break
#         file_count = file_count + 1
#
# for nf in neg_files:
#     with open(nf, "r", encoding='utf-8') as f:
#         indexCounter = 0
#         line = f.readline()
#         cleanedLine = cleanSentences(line)
#         split = cleanedLine.split()
#         for word in split:
#             try:
#                 ids[file_count][indexCounter] = wordsList.index(word)
#             except ValueError:
#                 ids[file_count][indexCounter] = 399999
#             indexCounter += 1
#             if indexCounter >= max_seq_num:
#                 break
#         file_count = file_count + 1
#
# # 保存到文件
# np.save('idsMatrix', ids)


from random import randint
batch_size = 24
lstm_units = 64
num_lables = 2
iterations = 100
lr = 0.001
ids = np.load('idsMatrix.npy')

# 辅助函数
def get_train_batch():
    lables = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(1, 11499)
            lables.append([1, 0])
        else:
            num = randint(13499, 24999)
            lables.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, lables

def get_test_batch():
    lables = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if (num <= 12499):
            lables.append([1, 0])
        else:
            lables.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, lables

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
tf.reset_default_graph()


labels = tf.placeholder(tf.float32, [batch_size, num_lables])
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num])
data = tf.Variable(
    tf.zeros([batch_size, max_seq_num, num_dimensions]), dtype=tf.float32
)
data = tf.nn.embedding_lookup(wordVectors, input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.5)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstm_units, num_lables]))
bias = tf.Variable(tf.constant(0.1, shape=[num_lables]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=prediction, labels=labels
))
optimizer = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)

saver = tf.compat.v1.train.Saver()

with tf.Session() as sess:
    if os.path.exists("models") and os.path.exists("models/che"):
        saver.restore(sess, tf.train.latest_checkpoint('models'))
    else:
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

    iterations = 1000000
    for step in range(iterations):
        next_batch, next_batch_labels = get_test_batch()
        if step % 50 == 0:
            print("step:", step, "正确率:", (sess.run(accuracy, {input_data:next_batch,labels:next_batch_labels}))* 100)
    if not os.path.exists("models"):
        os.mkdir("models")
    save_path = saver.save(sess, "models/model.ckpt")
    print("模型保存在：%s" % save_path)


















