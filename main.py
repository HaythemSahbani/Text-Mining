#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk

from time import time
t0 = time()


import preprocess
from feature_extraction import FeatureExtraction, PatternsFeatures
from nltk import NaiveBayesClassifier
from nltk.classify import MaxentClassifier
from sklearn.svm import LinearSVC

import random
import nltk.metrics


filename = "all_tweets.txt"

lines = preprocess.main(filename)
random.shuffle(lines)

all_tweets = [line[1] for line in lines]


ll = []
for line in lines:
    for lin in line[1]:
        ll.append(lin)

print len(ll)
print len(set(ll))
"""
hashtag_list = PatternsFeatures().get_most_frequent_pattern(PatternsFeatures().pattern_classifier(lines, '#'))
name_list = PatternsFeatures().get_most_frequent_pattern(PatternsFeatures().pattern_classifier(lines, '@'))
train_set_rate = int(len(lines)*0.75)
train_set, test_set = lines[:train_set_rate], lines[train_set_rate:]
all_tweets = [" ".join(line[1]) for line in train_set]

tfidf_accuracy_list_nb = []
tfidf_f_measure_list_nb = []
tfidf_accuracy_list_svm = []
tfidf_f_measure_list_svm = []
tfidf_accuracy_list_maxent = []
tfidf_f_measure_list_maxent = []


fd_accuracy_list_nb = []
fd_f_measure_list_nb = []
fd_accuracy_list_svm = []
fd_f_measure_list_svm = []
fd_accuracy_list_maxent = []
fd_f_measure_list_maxent = []



unigram_tfdf_ftr = FeatureExtraction(90)
unigram_fd_ftr = FeatureExtraction(90)
ftr = FeatureExtraction(90)
unigram_tfdf_ftr.tf_idf_features(all_tweets, n_grams=1)
unigram_fd_ftr.most_frequent_unigrams(" ".join(all_tweets))
ftr.most_frequent_bigrams(" ".join(all_tweets))
for hashtag in hashtag_list:
    unigram_tfdf_ftr.set_unigram_features_list(hashtag)
    unigram_fd_ftr.set_unigram_features_list(hashtag)

for name in name_list:
    unigram_tfdf_ftr.set_unigram_features_list(name)
    unigram_fd_ftr.set_unigram_features_list(name)



fd_unigram_featuresets_train = [(unigram_fd_ftr.unigram_features(line[1]), line[0]) for line in train_set]


classifier1 = NaiveBayesClassifier.train(fd_unigram_featuresets_train)


classifier1.show_most_informative_features()
print "###############################"


bigram_featuresets_test = [(ftr.bigram_features(line[1]), line[0]) for line in test_set]
bigram_featuresets_train = [(ftr.bigram_features(line[1]), line[0]) for line in train_set]


classifier1 = NaiveBayesClassifier.train(bigram_featuresets_train)
classifier1.show_most_informative_features()


print "################################################"
featuresets_train = bigram_featuresets_train + fd_unigram_featuresets_train
##############################################################################
classifier3 = NaiveBayesClassifier.train(featuresets_train)
classifier3.show_most_informative_features()

"""