#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/25 14:47
@Description: 
"""
import pickle
import os
__author__ = 'lch02'

test_path = r'/Users/lch/Desktop/pycharm/Emotion-Analyse/test'
f_path = r'E:\Repositories\Emotion-Analyse\features'

# 最好的特征值
best_words = []

# 单个词选择阈值
word_scores_threshold = 10000
# 双词搭配选择阈值
bigram_scores_threshold = 10000

# 分割数据的阈值
choose_threshold = 320000

