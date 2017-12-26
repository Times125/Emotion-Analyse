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

test_path = r'E:\Repositories\Emotion-Analyse\test'
f_path = r'E:\Repositories\Emotion-Analyse\features'
best_words = []
# 单个词选择阈值
word_scores_threshold = 10000
# 双词搭配选择阈值
bigram_scores_threshold = 10000

pos_data = pickle.load(open(os.path.join(test_path, 'pos_review.pkl'), 'rb'))
neg_data = pickle.load(open(os.path.join(test_path, 'neg_review.pkl'), 'rb'))
