#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/25 14:46
@Description: 处理xlxsw中的文本，将其中的数据（100w条）导出
"""
import os
import re
import nltk
import pickle
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk import regexp_tokenize
from nltk.corpus import stopwords
from config import *
from openpyxl import load_workbook

__author__ = 'lch02'

"""
将Excle中的数据导出
"""
def export_data():
    file_name = os.path.join(test_path, r'Sentiment Analysis Dataset.xlsx')
    wb = load_workbook(file_name, read_only=True)
    ws = wb.active
    neg = []
    pos = []
    for row in ws.iter_rows('A:B'):
        label = row[0].value
        content = row[1].value
        if content is not None:
            content = text_parse(content)
            print label, ' ', content
        else:
            continue
        if label == 0:
            neg.append((content, label))
        else:
            pos.append((content, label))
    neg_file = os.path.join(test_path, 'neg_pickle.pkl')
    pos_file = os.path.join(test_path, 'pos_pickle.pkl')
    with open(neg_file, 'wb') as f:
        pickle.dump(neg, f)
    with open(pos_file, 'wb') as f:
        pickle.dump(pos, f)


def text_parse(input_text, language='en'):
    sentence = input_text.strip().lower()
    sentence = re.sub(r'@\s*[\w]+ | ?#[\w]+ | ?&[\w]+; | ?[^\x00-\xFF]+', '', sentence)
    special_tag = set(
        ['.', ',', '#', '!', '(', ')', '*', '`', ':', '"', '‘', '’', '“', '”', '@', '：', '^', '/', ']', '[', ';', '=', '_'])
    pattern = r""" (?x)(?:[a-z]\.)+ 
                  | \d+(?:\.\d+)?%?\w+
                  | \w+(?:[-']\w+)*"""

    word_list = regexp_tokenize(sentence, pattern)
    if language == 'en':
        filter_word = [w for w in word_list if
                       w not in stopwords.words('english') and w not in special_tag]  # 去停用词和特殊标点符号
    elif language == 'fr':
        filter_word = [w for w in word_list if
                       w not in stopwords.words('french') and w not in special_tag]  # 去停用词和特殊标点符号
    res = ' '.join(filter_word)
    return res

"""
构建特征
"""
def bag_of_words(words):
    return dict([(word, True) for word in words])

"""
把所有词和双词搭配一起作为特征
"""
def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1500):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)
