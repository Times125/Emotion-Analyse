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

from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

__author__ = 'lch02'


def create_classifier(featx):

    pos_data = pickle.load(open(os.path.join(config.test_path, 'pos_review.pkl'), 'rb'))
    neg_data = pickle.load(open(os.path.join(config.test_path, 'neg_review.pkl'), 'rb'))

    pos_words = pos_data[config.choose_threshold:]
    neg_words = neg_data[config.choose_threshold:]

    print len(pos_words), '------', len(neg_words)
    pos_features = [(featx(w_lst), 'pos') for w_lst in pos_words]
    neg_features = [(featx(w_lst), 'neg') for w_lst in neg_words]

    negoff = int(len(neg_features) * 0.9)
    posoff = int(len(pos_features) * 0.9)

    r_pos_cut = pos_features[:posoff]
    r_neg_cut = neg_features[:negoff]

    print r_pos_cut is None, '---r_pos_cut----', len(r_pos_cut)
    print r_neg_cut is None, '---r_neg_cut----', len(r_neg_cut)

    t_pos_cut = pos_features[posoff:]
    t_neg_cut = neg_features[negoff:]

    print t_pos_cut is None, '---t_pos_cut----', len(t_pos_cut)
    print t_neg_cut is None, '---t_neg_cut----', len(t_neg_cut)

    r_pos_cut.extend(r_neg_cut)
    train_set = r_pos_cut
    t_pos_cut.extend(t_neg_cut)
    test_set = t_pos_cut

    # print pos_features
    print train_set is None, '---train_set----', len(train_set)
    print test_set is None, '-----test_set--', len(test_set)

    nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
    print "NBayes accuracy is %.7f" % nltk.classify.accuracy(nb_classifier, test_set)

    bernoulli_classifier = SklearnClassifier(BernoulliNB()).train(train_set)
    print "BernoulliNB accuracy is %.7f" % nltk.classify.accuracy(bernoulli_classifier, test_set)

    classifier_pkl = os.path.join(config.test_path, 'my_classifier.pkl')  # 消极语料
    with open(classifier_pkl, 'wb') as f:
        pickle.dump(nb_classifier, f)

    best_feats_pkl = os.path.join(config.test_path, 'best_feats.pkl')  # 消极语料
    with open(best_feats_pkl, 'wb') as f:
        pickle.dump(config.best_words, f)

    print 'done!'


def get_model():
    with open(os.path.join(config.test_path, 'my_classifier.pkl'), 'rb') as f:
        classifier = pickle.load(f)
    return classifier
