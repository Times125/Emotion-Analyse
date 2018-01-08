#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/25 14:32
@Description: 
"""
import json
import socket

from deal_files import export_data
from deal_features import *
from my_classifier import *
from multiprocessing import Pool
from openpyxl import load_workbook
from deal_files import text_parse
from data_filter import review_filter

__author__ = 'lch02'

"""
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

    bernoulli_classifier = SklearnClassifier(LinearSVC()).train(train_set)
    print "BernoulliNB accuracy is %.7f" % nltk.classify.accuracy(bernoulli_classifier, test_set)

    classifier_pkl = os.path.join(config.test_path, 'my_classifier_mod.pkl')
    with open(classifier_pkl, 'wb') as f:
        pickle.dump(bernoulli_classifier, f)

    nb_classifier_pkl = os.path.join(config.test_path, 'my_nb_classifier_mod.pkl')
    with open(nb_classifier_pkl, 'wb') as f:
        pickle.dump(nb_classifier, f)

    best_feats_pkl = os.path.join(config.test_path, 'best_feats_pkl_mod.pkl')
    with open(best_feats_pkl, 'wb') as f:
        pickle.dump(config.best_words, f)

    print 'test_my done!'


def deals_fun(cat, my_classifier, pos_words, neg_words):
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


def train_again():
    my_classifier = get_model()
    pos_data = pickle.load(open(os.path.join(config.test_path, 'pos_review.pkl'), 'rb'))
    neg_data = pickle.load(open(os.path.join(config.test_path, 'neg_review.pkl'), 'rb'))
    pos_words = pos_data[:]
    neg_words = neg_data[:]
    pool = Pool()
    for i in range(2):
        pool.apply_async(deals_fun, args=(i, my_classifier, pos_words, neg_words,))
    pool.close()
    pool.join()
    test_my(best_bigram_words_features)
"""

def test_classifier():
    file_name = os.path.join(test_path, 'testdata.xlsx')
    wb = load_workbook(file_name, read_only=True)
    ws = wb.active
    neg = []
    pos = []
    for row in ws.iter_rows('A:B'):
        label = row[0].value
        content = row[1].value
        if content is not None:
            content = text_parse(content)
            if len(content) == 0:
                continue
            elif label == 0 and len(content) != 0:
                neg.append((content, 0))
            elif label == 4 and len(content) != 0:
                pos.append((content, 4))
    pos.extend(neg)
    test = pos
    sum = len(test)
    print len(neg), '----', sum
    corrected = 0.0
    classifier = get_model()
    print 'model loaded!'
    for words, label in test:
        f_dict = best_bigram_words_features(words)
        cl = classifier.classify(f_dict)
        if label == 0 and cl == 'neg':
            corrected += 1
        elif label == 4 and cl == 'pos':
            corrected += 1
    print 'accuracy is %.7f' % (corrected / sum)

def main():
    classifier = get_model()
    print 'load success!'
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 7004))
    s.listen(10)
    while True:
        try:
            sock, addr = s.accept()
            data = sock.recv(102400)
            data = data.decode('utf-8').encode('utf-8')
            if data is 'auto':
                if len(config.best_words) == 0:
                    config.best_words = pickle.load(open(os.path.join(config.test_path, 'best_feats.pkl'), 'rb'))
                create_classifier(best_bigram_words_features)
                sock.send("complete auto!")
                sock.close()
                continue
            deal = best_words_features(text_parse(data))
            res = classifier.classify(deal)
            if res == 'pos':
                p = 1.0
            else:
                p = 0.0
            res_json = json.dumps(p).encode('utf-8')
            sock.send(res_json)
            sock.close()
        except KeyboardInterrupt:
            print 'Exit'
            exit()
        except:
            print 'Error'

if __name__ == '__main__':

    # main()
    # export_data()
    # review_filter()
    """
    word_bigram_score_dict = word_bigram_scores()
    config.best_words = get_best_words(word_bigram_score_dict, 5000)
    best_feats_pkl = os.path.join(config.test_path, 'best_feats.pkl')
    with open(best_feats_pkl, 'wb') as f:
        pickle.dump(config.best_words, f)
    create_classifier(best_bigram_words_features)
    """
    test_classifier() # NB 0.8189415 # SVM 0.5961003