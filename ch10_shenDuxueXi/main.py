# -*- coding: utf-8 -*-

"""
@Time    : 2019/9/16 14:33
@User    : HouYushan
@Author  : xueba1521
@FileName: main.py
@Software: PyCharm
@Blog    ：http://---
"""
import bp
import mnist_loader

net = bp.Network([784, 100, 10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print(type(test_data))
net.SGD(list(training_data), 30, 10, 3.0, test_data=list(test_data))