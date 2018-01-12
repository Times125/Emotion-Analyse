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
import collections
from nltk.classify import SklearnClassifier
from nltk.metrics import *
from sklearn.svm import LinearSVC

__author__ = 'lch02'


def create_classifier(featx):
    pos_data = pickle.load(open(os.path.join(config.test_path, 'pos_review.pkl'), 'r'))
    neg_data = pickle.load(open(os.path.join(config.test_path, 'neg_review.pkl'), 'r'))

    pos_words = pos_data[:]
    neg_words = neg_data[:]

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

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
    nba = nltk.classify.accuracy(nb_classifier, test_set)
    for i, (feats, label) in enumerate(test_set):
        refsets[label].add(i)
        observed = nb_classifier.classify(feats)
        testsets[observed].add(i)
    print "NBayes accuracy is %.7f" % nba  # 0.5325077

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    svm_classifier = SklearnClassifier(LinearSVC()).train(train_set)
    for i, (feats, label) in enumerate(test_set):
        refsets[label].add(i)
        observed = svm_classifier.classify(feats)
        testsets[observed].add(i)
    svmm = nltk.classify.accuracy(svm_classifier, test_set)
    print "SVM accuracy is %.7f" % svmm  # 0.6604747



    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    maxent_classifier = nltk.classify.MaxentClassifier.train(train_set, max_iter=7)
    for i, (feats, label) in enumerate(test_set):
        refsets[label].add(i)
        observed = maxent_classifier.classify(feats)
        testsets[observed].add(i)
    maxent = nltk.classify.accuracy(maxent_classifier, test_set)
    print "MaxentClassifier accuracy is %.7f" % maxent  # 0.6449948


    classifier_pkl = os.path.join(config.test_path, 'my_classifier_svm.pkl')  # 消极语料
    with open(classifier_pkl, 'wb') as f:
        pickle.dump(svm_classifier, f)

    classifier_pkl = os.path.join(config.test_path, 'my_classifier_maxent.pkl')  # 消极语料
    with open(classifier_pkl, 'wb') as f:
        pickle.dump(maxent_classifier, f)

    classifier_pkl = os.path.join(config.test_path, 'my_classifier_nb.pkl')  # 消极语料
    with open(classifier_pkl, 'wb') as f:
        pickle.dump(nb_classifier, f)

    print 'done!'


def get_model():
    with open(os.path.join(config.test_path, 'my_classifier_svm.pkl'), 'rb') as f:
        classifier = pickle.load(f)
    return classifier
