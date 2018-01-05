#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/26 9:51
@Description: 
"""
import itertools
import os
import pickle
import config
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from config import test_path
from nltk.probability import FreqDist, ConditionalFreqDist
__author__ = 'lch02'


"""
计算单个词和双词搭配的贡献（信息量
"""
def word_bigram_scores():
    pos_data = pickle.load(open(os.path.join(test_path, 'pos_review.pkl'), 'rb'))
    neg_data = pickle.load(open(os.path.join(test_path, 'neg_review.pkl'), 'rb'))

    pos_words = list(itertools.chain(*pos_data))
    neg_words = list(itertools.chain(*neg_data))

    pos_bigram_finder = BigramCollocationFinder.from_words(pos_words)
    neg_bigram_finder = BigramCollocationFinder.from_words(neg_words)

    pos_bigrams = pos_bigram_finder.nbest(BigramAssocMeasures.chi_sq, config.bigram_scores_threshold)
    neg_bigrams = neg_bigram_finder.nbest(BigramAssocMeasures.chi_sq, config.bigram_scores_threshold)

    pos_words.extend(pos_bigrams)
    neg_words.extend(neg_bigrams)

    word_tf = FreqDist()    # 统计所有词频
    con_word_tf = ConditionalFreqDist()     # 统计每个词的概率分布

    for word in pos_words:
        word_tf[word] += 1
        con_word_tf['pos'][word] += 1
    for word in neg_words:
        word_tf[word] += 1
        con_word_tf['neg'][word] += 1
    pos_word_count = con_word_tf['pos'].N()     # 积极词的数量
    neg_word_count = con_word_tf['neg'].N()     # 消极词的数量
    total_word_count = pos_word_count + neg_word_count      # 总词
    bigram_scores_dict = {}
    for word, freq in word_tf.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(con_word_tf['pos'][word], (freq, pos_word_count), total_word_count)   # 计算积极词的卡方统计量
        neg_score = BigramAssocMeasures.chi_sq(con_word_tf['neg'][word], (freq, neg_word_count), total_word_count)   # 计算消极词的卡方统计量
        bigram_scores_dict[word] = pos_score + neg_score
    return bigram_scores_dict

"""
选择贡献最大的特征
"""
def get_best_words(scores_dict, threshold=10000):
    best = sorted(scores_dict.iteritems(), key=lambda (word, score): score, reverse=True)[:threshold]   # 从大到小排列,选择前10000个
    best_words = set([w for w, s in best])
    return best_words

"""  
选择1：最有信息量的单个词作为特征
"""
def best_words_features(words):
    if config.best_words is None:
        config.best_words = pickle.load(open(os.path.join(config.test_path, 'best_feats.pkl'), 'rb'))
    lst = []
    for word in words:
        if word in config.best_words:
            lst.append((word, True))
        else:
            lst.append((word, False))
    return dict(lst)

"""
选择2：把所有词和双词搭配一起作为特征
"""
def best_bigram_words_features(words, score_fn=BigramAssocMeasures.chi_sq, n=1500):
    try:
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, n)
    except ZeroDivisionError:
        words.append(' ')
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_words_features(words))
    return d
