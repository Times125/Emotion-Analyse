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
    nat_data = pickle.load(open(os.path.join(config.test_path, 'nat_reviews_mod.pkl'), 'rb'))

    pos_len = int(len(pos_data) * 0.2)  # 2470
    neg_len = int(len(neg_data) * 0.15)  # 2740
    nat_len = int(len(nat_data) * 0.1)  # 2956
    a = [pos_len, neg_len, nat_len]
    a.sort()
    print a
    threshold = a[0]
    print '*********', threshold
    pos_words = pos_data[pos_len:]
    neg_words = neg_data[neg_len:]
    nat_words = nat_data[nat_len:]

    print len(pos_words), '------', len(neg_words)
    pos_features = [(featx(w_lst), 'pos') for w_lst in pos_words]
    neg_features = [(featx(w_lst), 'neg') for w_lst in neg_words]
    nat_features = [(featx(w_lst), 'nat') for w_lst in nat_words]

    negoff = int(len(neg_features) * 0.9)
    posoff = int(len(pos_features) * 0.9)
    natoff = int(len(nat_features) * 0.9)

    r_pos_cut = pos_features[:posoff]
    r_neg_cut = neg_features[:negoff]
    r_nat_cut = neg_features[:natoff]

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

    nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
    nba = nltk.classify.accuracy(nb_classifier, test_set)
    print "NBayes accuracy is %.7f" % nba

    bernoulli_classifier = SklearnClassifier(BernoulliNB()).train(train_set)
    bnla = nltk.classify.accuracy(bernoulli_classifier, test_set)
    print "BernoulliNB accuracy is %.7f" % bnla

    classifier_pkl = os.path.join(config.test_path, 'my_classifier_mod.pkl')
    with open(classifier_pkl, 'wb') as f:
        if nba > bnla:
            pickle.dump(nb_classifier, f)
            print 'nb_classifier'
        else:
            pickle.dump(bernoulli_classifier, f)
            print 'bernoulli_classifier'

    best_feats_pkl = os.path.join(config.test_path, 'best_feats_pkl_mod.pkl')
    with open(best_feats_pkl, 'wb') as f:
        pickle.dump(config.best_words, f)

    print 'test_my done!'


def deals_fun(cat, my_classifier, pos_words, neg_words, nat_words):
    neg_reviews = []
    pos_reviews = []
    nat_reviews = []
    print '**************----',len(pos_words)
    if cat == 1:
        for words in pos_words:
            res = my_classifier.classify(best_bigram_words_features(words))
            print res,'+++++++++++++='
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
    elif cat == 2:
        for words in nat_words:
            res = my_classifier.classify(best_bigram_words_features(words))
            if res == 'nat':
                nat_reviews.append(words)
        print 2
        with open(os.path.join(config.test_path, 'nat_reviews_mod.pkl'), 'wb') as f:
            pickle.dump(nat_reviews, f)
            print 'mod nat_reviews done!'

def train_again():
    my_classifier = get_model()
    pos_data = pickle.load(open(os.path.join(config.test_path, 'pos_review.pkl'), 'rb'))
    neg_data = pickle.load(open(os.path.join(config.test_path, 'neg_review.pkl'), 'rb'))
    nat_data = pickle.load(open(os.path.join(config.test_path, 'nat_review.pkl'), 'rb'))
    pos_words = pos_data[:]
    neg_words = neg_data[:]
    nat_words = nat_data[:]
    # 2801 5132 7732
    pool = Pool()
    for i in range(3):
        pool.apply_async(deals_fun, args=(i, my_classifier, pos_words, neg_words, nat_words,))
    pool.close()
    pool.join()
    test_my(best_bigram_words_features)


if __name__ == '__main__':
    # export_data()
    word_bigram_score_dict = word_bigram_scores()
    config.best_words = get_best_words(word_bigram_score_dict, 10000)
    create_classifier(best_bigram_words_features)
    # train_again()