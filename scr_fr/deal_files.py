#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/25 14:46
@Description: 处理xlxsw中的文本，将其中的数据（160w条）导出
"""
import os
import re
import pickle

from nltk import regexp_tokenize
from nltk.corpus import stopwords
from config import test_path
from openpyxl import load_workbook
from multiprocessing import Pool
import config

__author__ = 'lch02'


def export_data():
    neg = []
    pos = []
    pos_data = pickle.load(open(os.path.join(config.test_path, 'pos_reviews_fr_2.pkl'), 'rb'))
    neg_data = pickle.load(open(os.path.join(config.test_path, 'neg_reviews_fr_2.pkl'), 'rb'))

    for row in neg_data:
        content = row
        if content is not None:
            content = text_parse(content)
            if len(content) == 0:
                continue
            elif len(content) != 0:
                neg.append(content)
    neg_file = os.path.join(test_path, 'neg_review.pkl')  # 消极语料
    with open(neg_file, 'wb') as f:
        pickle.dump(neg, f)

    for row in pos_data:
        content = row
        if content is not None:
            content = text_parse(content)
            if len(content) == 0:
                continue
            elif len(content) != 0:
                pos.append(content)
    pos_file = os.path.join(test_path, 'pos_review.pkl')  # 积极语料
    with open(pos_file, 'wb') as f:
        pickle.dump(pos, f)



"""
文本处理：取词、去停用词等
"""


def text_parse(input_text, language='fr'):
    sentence = input_text.strip().lower()
    sentence = re.sub(r'@\s*[\w]+ | ?#[\w]+ | ?&[\w]+; | ?[^\x00-\xFF]+', '', sentence)
    special_tag = set(
        ['.', ',', '#', '!', '(', ')', '*', '`', ':', '"', '‘', '’', '“', '”', '@', '：', '^', '/', ']', '[', ';', '=',
         '_'])
    pattern = r""" (?x)(?:[a-z]\.)+ 
                  | \d+(?:\.\d+)?%?\w+
                  | \w+(?:[-']\w+)*"""

    word_list = regexp_tokenize(sentence, pattern)
    if language == 'fr':
        filter_word = [w for w in word_list if
                       w not in stopwords.words('french') and w not in special_tag]  # 去停用词和特殊标点符号
    return filter_word
