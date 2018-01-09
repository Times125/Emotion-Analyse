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

    pos_data = pickle.load(open(os.path.join(config.test_path, 'pos_review.pkl'), 'rb'))
    neg_data = pickle.load(open(os.path.join(config.test_path, 'neg_review.pkl'), 'rb'))
    nat_data = pickle.load(open(os.path.join(config.test_path, 'nat_review.pkl'), 'rb'))

    pos_words = pos_data[config.choose_threshold:]
    neg_words = neg_data[config.choose_threshold*2:]
    nat_words = nat_data[config.choose_threshold*3:]

    print len(pos_words), '------', len(neg_words), '------', len(nat_words)
    pos_features = [(featx(w_lst), 'pos') for w_lst in pos_words]
    neg_features = [(featx(w_lst), 'neg') for w_lst in neg_words]
    nat_features = [(featx(w_lst), 'nat') for w_lst in nat_words]

    negoff = int(len(neg_features) * 0.9)
    posoff = int(len(pos_features) * 0.9)
    natoff = int(len(nat_features) * 0.9)

    r_pos_cut = pos_features[:posoff]
    r_neg_cut = neg_features[:negoff]
    r_nat_cut = nat_features[:natoff]

    print r_pos_cut is None, '---r_pos_cut----', len(r_pos_cut)
    print r_neg_cut is None, '---r_neg_cut----', len(r_neg_cut)
    print r_nat_cut is None, '---r_nat_cut----', len(r_nat_cut)

    t_pos_cut = pos_features[posoff:]
    t_neg_cut = neg_features[negoff:]
    t_nat_cut = nat_features[natoff:]

    print t_pos_cut is None, '---t_pos_cut----', len(t_pos_cut)
    print t_neg_cut is None, '---t_neg_cut----', len(t_neg_cut)
    print t_nat_cut is None, '---t_nat_cut----', len(t_nat_cut)

    r_pos_cut.extend(r_neg_cut)
    r_pos_cut.extend(r_nat_cut)
    train_set = r_pos_cut

    t_pos_cut.extend(t_neg_cut)
    t_pos_cut.extend(t_nat_cut)
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
    print "NBayes accuracy is %.7f" % nba # 0.5325077
    print "NBayes pos precision is ：", precision(refsets['pos'], testsets['pos']) # 0.33064516129
    print "NBayes neg precision is ：", precision(refsets['neg'], testsets['neg']) # 0.580487804878
    print "NBayes nat precision is ：", precision(refsets['nat'], testsets['nat']) # 0.698979591837
    print "NBayes pos recall is :", recall(refsets['pos'], testsets['pos'])  # 0.67955801105
    print "NBayes neg recall is :", recall(refsets['neg'], testsets['neg'])  # 0.37898089172
    print "NBayes nat recall is :", recall(refsets['nat'], testsets['nat'])  # 0.57805907173

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    svm_classifier = SklearnClassifier(LinearSVC()).train(train_set)
    for i, (feats, label) in enumerate(test_set):
        refsets[label].add(i)
        observed = svm_classifier.classify(feats)
        testsets[observed].add(i)
    svmm = nltk.classify.accuracy(svm_classifier, test_set)
    print "SVM accuracy is %.7f" % svmm  # 0.6594427
    print "SVM pos precision is ：", precision(refsets['pos'], testsets['pos'])  # 0.753846153846
    print "SVM neg precision is ：", precision(refsets['neg'], testsets['neg'])  # 0.648648648649
    print "SVM nat precision is ：", precision(refsets['nat'], testsets['nat'])  # 0.643435980551
    print "SVM pos recall is :", recall(refsets['pos'], testsets['pos'])  # 0.541436464088
    print "SVM neg recall is :", recall(refsets['neg'], testsets['neg'])  # 0.458598726115
    print "SVM nat recall is :", recall(refsets['nat'], testsets['nat'])  # 0.837552742616

    maxent_classifier = nltk.classify.MaxentClassifier.train(train_set, max_iter=10)
    maxent = nltk.classify.accuracy(maxent_classifier, test_set)
    print "MaxentClassifier accuracy is %.7f" % maxent



    """
    classifier_pkl = os.path.join(config.test_path, 'my_classifier.pkl')  # 消极语料
    with open(classifier_pkl, 'wb') as f:
        if nba > bnla:
            pickle.dump(nb_classifier, f)
        else:
            pickle.dump(bernoulli_classifier, f)

    best_feats_pkl = os.path.join(config.test_path, 'best_feats.pkl')  # 消极语料
    with open(best_feats_pkl, 'wb') as f:
        pickle.dump(config.best_words, f)
    """
    print 'done!'


def get_model():
    with open(os.path.join(config.test_path, 'my_classifier.pkl'), 'rb') as f:
        classifier = pickle.load(f)
    return classifier
