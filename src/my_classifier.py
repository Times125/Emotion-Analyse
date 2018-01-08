#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/26 18:23
@Description: 
"""
import config
import os
import nltk
import cPickle as pickle
# import pickle

from nltk.classify import SklearnClassifier
from sklearn.svm import LinearSVC

__author__ = 'lch02'


def create_classifier(featx):

    pos_data = pickle.load(open(os.path.join(config.test_path, 'pos_review.pkl'), 'rb'))
    neg_data = pickle.load(open(os.path.join(config.test_path, 'neg_review.pkl'), 'rb'))

    pos_dataun = pickle.load(open(os.path.join(config.test_path, 'pos_reviews.pkl'), 'rb'))
    neg_dataun = pickle.load(open(os.path.join(config.test_path, 'neg_reviews.pkl'), 'rb'))
    pposun = pos_dataun[config.choose_threshold:]
    pnegun = neg_dataun[config.choose_threshold:]

    pos_words = pos_data[config.choose_threshold:]
    neg_words = neg_data[config.choose_threshold:]

    print len(pos_words), '------', len(neg_words)
    pos_features = [(featx(w_lst), 'pos') for w_lst in pos_words]
    neg_features = [(featx(w_lst), 'neg') for w_lst in neg_words]

    negoff = int(len(neg_features) * 0.9)
    posoff = int(len(pos_features) * 0.9)

    aposun = pos_dataun[:posoff]
    anegun = neg_dataun[:negoff]

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

    """
    保存训练和测试数据
    """
    with open(os.path.join(config.test_path, 'train_data.pkl'), 'wb') as f:
        pposun.extend(pnegun)
        pickle.dump(pposun, f)
        print 'better neg_reviews done!'
    with open(os.path.join(config.test_path, 'test_data.pkl'), 'wb') as f:
        aposun.extend(anegun)
        pickle.dump(aposun, f)
        print 'better neg_reviews_lst done!'

    """
    训练两个分类器
    """
    nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
    nba = nltk.classify.accuracy(nb_classifier, test_set)
    print "NBayes accuracy is %.7f" % nba  # 0.93993803

    svm_classifier = SklearnClassifier(LinearSVC()).train(train_set)
    svmm = nltk.classify.accuracy(svm_classifier, test_set)
    print "svm_classifier accuracy is %.7f" % svmm  # 90.498

    """
    保存准确率更大的那个模型
    """
    classifier_pkl = os.path.join(config.test_path, 'my_classifier.pkl')
    with open(classifier_pkl, 'wb') as f:
        if nba > svmm:
            pickle.dump(nb_classifier, f)
            print 'NBayes'
        else:
            pickle.dump(svm_classifier, f)
            print 'SVM'
    classifier_pkl_1 = os.path.join(config.test_path, 'my_classifier_svm.pkl')
    with open(classifier_pkl_1, 'wb') as f:
        pickle.dump(svm_classifier, f)
        print 'SVM'
    print 'done!'


def get_model():
    with open(os.path.join(config.test_path, 'my_classifier_svm.pkl'), 'rb') as f:
        classifier = pickle.load(f)
    return classifier
