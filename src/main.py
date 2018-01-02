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
from multiprocessing import Pool

__author__ = 'lch02'


def test_my(featx):

    pos_data = pickle.load(open(os.path.join(config.test_path, 'pos_reviews_mod.pkl'), 'rb'))
    neg_data = pickle.load(open(os.path.join(config.test_path, 'neg_reviews_mod.pkl'), 'rb'))
    pos_len = int(len(pos_data) * 0.5)
    neg_len = int(len(neg_data) * 0.5)
    threshold = (lambda x, y: y if x > y else x)(pos_len, neg_len)

    pos_words = pos_data[threshold:]
    neg_words = neg_data[threshold:]

    print len(pos_words), '------', len(neg_words)
    pos_features = [(featx(w_lst), 'pos') for w_lst in pos_words]
    neg_features = [(featx(w_lst), 'neg') for w_lst in neg_words]

    negoff = int(len(neg_features) * 0.9)
    posoff = int(len(pos_features) * 0.9)

    r_pos_cut = pos_features[:posoff]
    r_neg_cut = neg_features[:negoff]

    t_pos_cut = pos_features[posoff:]
    t_neg_cut = neg_features[negoff:]

    r_pos_cut.extend(r_neg_cut)
    train_set = r_pos_cut
    t_pos_cut.extend(t_neg_cut)
    test_set = t_pos_cut

    nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
    print "NBayes accuracy is %.7f" % nltk.classify.accuracy(nb_classifier, test_set)

    bernoulli_classifier = SklearnClassifier(BernoulliNB()).train(train_set)
    print "BernoulliNB accuracy is %.7f" % nltk.classify.accuracy(bernoulli_classifier, test_set)

    classifier_pkl = os.path.join(config.test_path, 'my_classifier_mod.pkl')  # 消极语料
    with open(classifier_pkl, 'wb') as f:
        pickle.dump(bernoulli_classifier, f)

    nb_classifier_pkl = os.path.join(config.test_path, 'my_nb_classifier_mod.pkl')  # 消极语料
    with open(nb_classifier_pkl, 'wb') as f:
        pickle.dump(nb_classifier, f)

    best_feats_pkl = os.path.join(config.test_path, 'best_feats_pkl_mod.pkl')  # 消极语料
    with open(best_feats_pkl, 'wb') as f:
        pickle.dump(config.best_words, f)

    print 'test_my done!'


def deals_fun(cat):
    neg_reviews = []
    pos_reviews = []
    if cat == 1:
        for words in pos_words:
            res = my_classifier.classify(best_bigram_words_features(words))
            if res == 'pos':
                pos_reviews.append(words)
        print 1
        with open(os.path.join(config.test_path, 'pos_reviews_mod.pkl'), 'wb') as f:
            pickle.dump(pos_reviews, f)
            print 'mod pos_reviews done!'
    elif cat == 0:
        for words in neg_words:
            res = my_classifier.classify(best_bigram_words_features(words))
            if res == 'neg':
                neg_reviews.append(words)
        print 0
        with open(os.path.join(config.test_path, 'neg_reviews_mod.pkl'), 'wb') as f:
            pickle.dump(neg_reviews, f)
            print 'mod neg_reviews done!'





    """
    scores_dict = word_scores()
    config.best_words = get_best_words(scores_dict, 20000)
    create_classifier(best_words_features)
    """
if __name__ == '__main__':
    # export_data()
    #word_bigram_score_dict = word_bigram_scores()
    #config.best_words = get_best_words(word_bigram_score_dict, 5000)
    #create_classifier(best_bigram_words_features)

    my_classifier = get_model()
    pos_data = pickle.load(open(os.path.join(config.test_path, 'pos_review.pkl'), 'rb'))
    neg_data = pickle.load(open(os.path.join(config.test_path, 'neg_review.pkl'), 'rb'))
    pos_words = pos_data[:130000]
    neg_words = neg_data[:130000]
    # pos_features = [best_bigram_words_features(w_lst) for w_lst in pos_words]
    # neg_features = [best_bigram_words_features(w_lst) for w_lst in neg_words]
    pool = Pool()
    for i in range(2):
        pool.apply_async(deals_fun, args=(i,))
    pool.close()
    pool.join()
    test_my(best_bigram_words_features)