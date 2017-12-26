#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/25 14:32
@Description: 
"""
from deal_files import export_data
from deal_features import *
from my_classifier import *

__author__ = 'lch02'

if __name__ == '__main__':
    # export_data()
    scores_dict = word_scores()
    config.best_words = get_best_words(scores_dict, 100)
    print config.best_words
    # create_classifier(best_bigram_words_features)
    create_classifier(best_words_features)
