#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/26 18:23
@Description: 
"""
import config
import os
import itertools
import nltk
import pickle

__author__ = 'lch02'


def create_classifier(featx):
    pos_words = list(itertools.chain(*(config.pos_data[300000:])))
    neg_words = list(itertools.chain(*(config.neg_data[300000:])))
    print len(pos_words),'------',len(neg_words)
    pos_features = [(featx(w_lst), 'pos') for w_lst in pos_words]
    neg_features = [(featx(w_lst), 'neg') for w_lst in neg_words]

    negoff = int(len(neg_features) * 0.8)
    posoff = int(len(pos_features) * 0.8)

    r_pos_cut = pos_features[:posoff]
    r_neg_cut = neg_features[:negoff]

    t_pos_cut = pos_features[posoff:]
    t_neg_cut = neg_features[negoff:]

    train_set = r_pos_cut.extend(r_neg_cut)
    test_set = t_pos_cut.extend(t_neg_cut)

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print nltk.classify.accuracy(classifier, test_set)
    classifier_pkl = os.path.join(config.test_path, 'my_classifier.pkl')  # 消极语料
    with open(classifier_pkl, 'wb') as f:
        pickle.dump(classifier, f)
    print 'done!'
