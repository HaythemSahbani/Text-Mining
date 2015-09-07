__author__ = 'Haythem Sahbani'

######################################
#
# This file contains the evaluation
# methods for the classification models.
# This process takes several hours.
#
#######################################

import preprocess
from feature_extraction import FeatureExtraction, PatternsFeatures
from nltk import NaiveBayesClassifier
from nltk.classify import MaxentClassifier
from sklearn.svm import LinearSVC

import random
import collections
import nltk.metrics
from time import time
from matplotlib import pyplot as plt


def unigram_evaluation(lines):
    """
    + plots the classification F1-measure using unigrams
    + prints a table containing the max accuracy and F1-measure obtained and the number of feature reached at
    :param lines: list of tweets
    :return:
    """

    random.shuffle(lines)

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

    for i in range(10, 200, 20):

        unigram_tfdf_ftr = FeatureExtraction(i)
        unigram_fd_ftr = FeatureExtraction(i)

        unigram_tfdf_ftr.tf_idf_features(all_tweets, n_grams=1)
        unigram_fd_ftr.most_frequent_unigrams(" ".join(all_tweets))

        for hashtag in hashtag_list:
            unigram_tfdf_ftr.set_unigram_features_list(hashtag)
            unigram_fd_ftr.set_unigram_features_list(hashtag)

        for name in name_list:
            unigram_tfdf_ftr.set_unigram_features_list(name)
            unigram_fd_ftr.set_unigram_features_list(name)

        tfidf_unigram_featuresets_test = [(unigram_tfdf_ftr.unigram_features(line[1]), line[0]) for line in test_set]
        tfidf_unigram_featuresets_train = [(unigram_tfdf_ftr.unigram_features(line[1]), line[0]) for line in train_set]

        fd_unigram_featuresets_test = [(unigram_fd_ftr.unigram_features(line[1]), line[0]) for line in test_set]
        fd_unigram_featuresets_train = [(unigram_fd_ftr.unigram_features(line[1]), line[0]) for line in train_set]


###############################################################################################

        classifier1 = NaiveBayesClassifier.train(tfidf_unigram_featuresets_train)
        classifier2 = MaxentClassifier.train(tfidf_unigram_featuresets_train)
        classifier3 = nltk.classify.SklearnClassifier(LinearSVC())
        classifier3.train(tfidf_unigram_featuresets_train)



        refsets = collections.defaultdict(set)
        testsets1 = collections.defaultdict(set)
        testsets2 = collections.defaultdict(set)
        testsets3 = collections.defaultdict(set)

        for i, (feats, label) in enumerate(tfidf_unigram_featuresets_test):
            refsets[label].add(i)
            observed1 = classifier1.classify(feats)
            observed2 = classifier2.classify(feats)
            observed3 = classifier3.classify(feats)
            testsets1[observed1].add(i)
            testsets2[observed2].add(i)
            testsets3[observed3].add(i)

        tfidf_accuracy_list_nb.append(nltk.classify.accuracy(classifier1, tfidf_unigram_featuresets_test))
        tfidf_f_measure_list_nb.append(nltk.metrics.f_measure(refsets['not'], testsets1['not']))
        tfidf_accuracy_list_svm.append(nltk.classify.accuracy(classifier3, tfidf_unigram_featuresets_test))
        tfidf_f_measure_list_svm.append(nltk.metrics.f_measure(refsets['not'], testsets3['not']))
        tfidf_accuracy_list_maxent.append(nltk.classify.accuracy(classifier2, tfidf_unigram_featuresets_test))
        tfidf_f_measure_list_maxent.append(nltk.metrics.f_measure(refsets['not'], testsets2['not']))

#########################################################################################################################



        classifier1 = NaiveBayesClassifier.train(tfidf_unigram_featuresets_train)
        classifier2 = MaxentClassifier.train(fd_unigram_featuresets_train)
        classifier3 = nltk.classify.SklearnClassifier(LinearSVC())
        classifier3.train(fd_unigram_featuresets_train)


        refsets = collections.defaultdict(set)
        testsets1 = collections.defaultdict(set)
        testsets2 = collections.defaultdict(set)
        testsets3 = collections.defaultdict(set)

        for i, (feats, label) in enumerate(fd_unigram_featuresets_test):
            refsets[label].add(i)
            observed1 = classifier1.classify(feats)
            observed2 = classifier2.classify(feats)
            observed3 = classifier3.classify(feats)
            testsets1[observed1].add(i)
            testsets2[observed2].add(i)
            testsets3[observed3].add(i)

        fd_accuracy_list_nb.append(nltk.classify.accuracy(classifier1, fd_unigram_featuresets_test))
        fd_f_measure_list_nb.append(nltk.metrics.f_measure(refsets['not'], testsets1['not']))
        fd_accuracy_list_svm.append(nltk.classify.accuracy(classifier3, fd_unigram_featuresets_test))
        fd_f_measure_list_svm.append(nltk.metrics.f_measure(refsets['not'], testsets3['not']))
        fd_accuracy_list_maxent.append(nltk.classify.accuracy(classifier2, fd_unigram_featuresets_test))
        fd_f_measure_list_maxent.append(nltk.metrics.f_measure(refsets['not'], testsets2['not']))


 ################################################################################
    print "+-----------------------------------------------------------------+"
    print "\t\t\t\t\tfrequency distribution classification measurements"
    print "+-----------------------------------------------------------------+"
    print "Naive Bayes\t\t\t\t\t %f \t\t\t\t%d" % (max(fd_accuracy_list_nb), (fd_accuracy_list_nb.index(max(fd_accuracy_list_nb))*20)+10)
    print "Maximum entropy\t\t\t\t %f \t\t\t\t%d" % (max(fd_accuracy_list_maxent), (fd_accuracy_list_maxent.index(max(fd_accuracy_list_maxent))*20)+10)
    print "Support Vector Machine\t\t %f \t\t\t\t%d" % (max(fd_accuracy_list_svm), (fd_accuracy_list_svm.index(max(fd_accuracy_list_svm))*20)+10)
    print "+-----------------------------------------------------------------+"
    print "+-----------------------------------------------------------------+"
    print "\t\t\t\t\t\t\tmax f-measure \t number of features "
    print "Naive Bayes\t\t\t\t\t %f \t\t\t\t%d" % (max(fd_f_measure_list_nb), (fd_f_measure_list_nb.index(max(fd_f_measure_list_nb))*20)+10)
    print "Maximum entropy\t\t\t\t %f \t\t\t\t%d" % (max(fd_f_measure_list_maxent), (fd_f_measure_list_maxent.index(max(fd_f_measure_list_maxent))*20)+10)
    print "Support Vector Machine\t\t %f \t\t\t\t%d" % (max(fd_f_measure_list_svm), (fd_f_measure_list_svm.index(max(fd_f_measure_list_svm))*20)+10)
    print "+-----------------------------------------------------------------+"

 ################################################################################
    print "+-----------------------------------------------------------------+"
    print "\t\t\t\t\ttf-idf classification measurements"
    print "+-----------------------------------------------------------------+"
    print "Naive Bayes\t\t\t\t\t %f \t\t\t\t%d" % (max(tfidf_accuracy_list_nb), (tfidf_accuracy_list_nb.index(max(tfidf_accuracy_list_nb))*20)+10)
    print "Maximum entropy\t\t\t\t %f \t\t\t\t%d" % (max(tfidf_accuracy_list_maxent), (tfidf_accuracy_list_maxent.index(max(tfidf_accuracy_list_maxent))*20)+10)
    print "Support Vector Machine\t\t %f \t\t\t\t%d" % (max(tfidf_accuracy_list_svm), (tfidf_accuracy_list_svm.index(max(tfidf_accuracy_list_svm))*20)+10)
    print "+-----------------------------------------------------------------+"
    print "+-----------------------------------------------------------------+"
    print "\t\t\t\t\t\t\tmax f-measure \t number of features "
    print "Naive Bayes\t\t\t\t\t %f \t\t\t\t%d" % (max(tfidf_f_measure_list_nb), (tfidf_f_measure_list_nb.index(max(tfidf_f_measure_list_nb))*20)+10)
    print "Maximum entropy\t\t\t\t %f \t\t\t\t%d" % (max(tfidf_f_measure_list_maxent), (tfidf_f_measure_list_maxent.index(max(tfidf_f_measure_list_maxent))*20)+10)
    print "Support Vector Machine\t\t %f \t\t\t\t%d" % (max(tfidf_f_measure_list_svm), (tfidf_f_measure_list_svm.index(max(tfidf_f_measure_list_svm))*20)+10)
    print "+-----------------------------------------------------------------+"
################################################################################

    print " time taken for the classification process %f sec " % (time() - t0)
######################################################################################################


    x_axis = [i for i in range(10, 200, 20)]
    plt.figure(facecolor='white')
    # fig1, = plt.plot(x_axis, tfidf_accuracy_list_nb, 'go-', label='Naive bayes accuracy')
    fig2, = plt.plot(x_axis, tfidf_f_measure_list_nb, 'ro-', label='Naive bayes f-measure')
    # fig3, = plt.plot(x_axis, tfidf_accuracy_list_svm, 'g*-', label='SVM accuracy')
    fig4, = plt.plot(x_axis, tfidf_f_measure_list_svm, 'go-', label='SVM f-measure')
    # fig5, = plt.plot(x_axis, tfidf_accuracy_list_maxent, 'g^-', label='max Entropy accuracy')
    fig6, = plt.plot(x_axis, tfidf_f_measure_list_maxent, 'o-', label='max Entropy f-measure')

    plt.xlabel('Number of features')
    plt.ylabel('Results')
    plt.title('Results of the classification using tf-idf')
    plt.legend(handles=[ fig2, fig4, fig6], loc=4)



##################################################################################
    plt.figure(facecolor='white')
    # fig1, = plt.plot(x_axis, fd_accuracy_list_nb, 'go-', label='Naive bayes accuracy')
    fig2, = plt.plot(x_axis, fd_f_measure_list_nb, 'ro-', label='Naive bayes f-measure')
    # fig3, = plt.plot(x_axis, fd_accuracy_list_svm, 'g*-', label='SVM accuracy')
    fig4, = plt.plot(x_axis, fd_f_measure_list_svm, 'go-', label='SVM f-measure')
    # fig5, = plt.plot(x_axis, fd_accuracy_list_maxent, 'g^-', label='max Entropy accuracy')
    fig6, = plt.plot(x_axis, fd_f_measure_list_maxent, 'o-', label='max Entropy f-measure')

    plt.xlabel('Number of features')
    plt.ylabel('Results')
    plt.title('Results of the classification using frequency distribution')
    plt.legend(handles=[fig2, fig4, fig6], loc=4)
    plt.show()


def bigram_evaluation(lines):
    """
    + plots the classification F1-measure using bigrams
    + prints a table containing the max accuracy and F1-measure obtained and the number of feature reached at
    :param lines: list of tweets
    :return:
    """
    random.shuffle(lines)
    train_set_rate = int(len(lines)*0.75)

    train_set, test_set = lines[:train_set_rate], lines[train_set_rate:]
    all_tweets = [" ".join(line[1]) for line in train_set]


    accuracy_list_nb = []
    f_measure_list_nb = []
    accuracy_list_svm = []
    f_measure_list_svm = []
    accuracy_list_maxent = []
    f_measure_list_maxent = []

    for i in range(10, 200, 20):

        ftr = FeatureExtraction(i)
        ftr.most_frequent_bigrams(" ".join(all_tweets))
        bigram_featuresets_test = [(ftr.bigram_features(line[1]), line[0]) for line in test_set]
        bigram_featuresets_train = [(ftr.bigram_features(line[1]), line[0]) for line in train_set]


        classifier1 = NaiveBayesClassifier.train(bigram_featuresets_train)
        classifier2 = MaxentClassifier.train(bigram_featuresets_train)
        classifier3 = nltk.classify.SklearnClassifier(LinearSVC())
        classifier3.train(bigram_featuresets_train)


        refsets = collections.defaultdict(set)
        testsets1 = collections.defaultdict(set)
        testsets2 = collections.defaultdict(set)
        testsets3 = collections.defaultdict(set)

        for i, (feats, label) in enumerate(bigram_featuresets_test):
            refsets[label].add(i)
            observed1 = classifier1.classify(feats)
            observed2 = classifier2.classify(feats)
            observed3 = classifier3.classify(feats)
            testsets1[observed1].add(i)
            testsets2[observed2].add(i)
            testsets3[observed3].add(i)

        # classifier.show_most_informative_features()


        accuracy_list_nb.append(nltk.classify.accuracy(classifier1, bigram_featuresets_test))
        f_measure_list_nb.append(nltk.metrics.f_measure(refsets['not'], testsets1['not']))
        accuracy_list_svm.append(nltk.classify.accuracy(classifier3, bigram_featuresets_test))
        f_measure_list_svm.append(nltk.metrics.f_measure(refsets['not'], testsets3['not']))
        accuracy_list_maxent.append(nltk.classify.accuracy(classifier2, bigram_featuresets_test))
        f_measure_list_maxent.append(nltk.metrics.f_measure(refsets['not'], testsets2['not']))
    ############################################################################################################"


    
    print "+-----------------------------------------------------------------+"
    print "\t\t\t\t\tbigram classification measurements"
    print "+-----------------------------------------------------------------+"
    print "\t\t\t\t\t\t\tmax accuracy \t number of features "
    print "Naive Bayes\t\t\t\t\t %f \t\t\t\t%d" % (max(accuracy_list_nb), (accuracy_list_nb.index(max(accuracy_list_nb))*20)+10)
    print "Maximum entropy\t\t\t\t %f \t\t\t\t%d" % (max(accuracy_list_maxent), (accuracy_list_maxent.index(max(accuracy_list_maxent))*20)+10)
    print "Support Vector Machine\t\t %f \t\t\t\t%d" % (max(accuracy_list_svm), (accuracy_list_svm.index(max(accuracy_list_svm))*20)+10)
    print "+-----------------------------------------------------------------+"
    print "+-----------------------------------------------------------------+"
    print "\t\t\t\t\t\t\tmax f-measure \t number of features "
    print "Naive Bayes\t\t\t\t\t %f \t\t\t\t%d" % (max(f_measure_list_nb), (f_measure_list_nb.index(max(f_measure_list_nb))*20)+10)
    print "Maximum entropy\t\t\t\t %f \t\t\t\t%d" % (max(f_measure_list_maxent), (f_measure_list_maxent.index(max(f_measure_list_maxent))*20)+10)
    print "Support Vector Machine\t\t %f \t\t\t\t%d" % (max(f_measure_list_svm), (f_measure_list_svm.index(max(f_measure_list_svm))*20)+10)
    print "+-----------------------------------------------------------------+"
################################################################################

    print " time taken for the classification process %f sec " % (time() - t0)
##################################################################################
    x_axis = [i for i in range(10, 200, 20)]

    plt.figure(facecolor='white')
    # fig1, = plt.plot(x_axis, accuracy_list_nb, 'r*-', label='Naive bayes accuracy')
    fig2, = plt.plot(x_axis, f_measure_list_nb, 'ro-', label='Naive bayes f-measure')
    # fig3, = plt.plot(x_axis, accuracy_list_svm, 'g*-', label='SVM accuracy')
    fig4, = plt.plot(x_axis, f_measure_list_svm, 'go-', label='SVM f-measure')
    # fig5, = plt.plot(x_axis, accuracy_list_maxent, '*-', label='max Entropy accuracy')
    fig6, = plt.plot(x_axis, f_measure_list_maxent, 'o-', label='max Entropy f-measure')

    plt.xlabel('Number of features')
    plt.ylabel('Results')
    plt.title('Results of the classification using bigrams')
    plt.legend(handles=[fig2, fig4, fig6], loc=4)
    plt.show()


def uni_and_bi_validation(lines):
    """
    + plots the classification F1-measure using bigrams and unigrams
    + prints a table containing the max accuracy and F1-measure obtained and the number of feature reached at
    :param lines: list of tweets
    :return:
    """
    accuracy_list_nb = []
    f_measure_list_nb = []
    accuracy_list_svm = []
    f_measure_list_svm = []
    accuracy_list_maxent = []
    f_measure_list_maxent = []


    random.shuffle(lines)

    hashtag_list = PatternsFeatures().get_most_frequent_pattern(PatternsFeatures().pattern_classifier(lines, '#'))
    name_list = PatternsFeatures().get_most_frequent_pattern(PatternsFeatures().pattern_classifier(lines, '@'))

    train_set_rate = int(len(lines)*0.75)
    train_set, test_set = lines[:train_set_rate], lines[train_set_rate:]
    all_tweets = " ".join([" ".join(line[1]) for line in train_set])

    ftr2 = FeatureExtraction(20)
    ftr2.most_frequent_bigrams(all_tweets)

    bigram_featuresets_test = [(ftr2.bigram_features(line[1]), line[0]) for line in test_set]
    bigram_featuresets_train = [(ftr2.bigram_features(line[1]), line[0]) for line in train_set]



    for i in range(10, 200, 20):
        ftr = FeatureExtraction(i)

        ftr.most_frequent_unigrams(all_tweets)

        for hashtag in hashtag_list:
            ftr.set_unigram_features_list(hashtag)
        for name in name_list:
            ftr.set_unigram_features_list(name)


        unigram_featuresets_test = [(ftr.unigram_features(line[1]), line[0]) for line in test_set]
        unigram_featuresets_train = [(ftr.unigram_features(line[1]), line[0]) for line in train_set]

        featuresets_test = bigram_featuresets_test + unigram_featuresets_test
        featuresets_train = bigram_featuresets_train + unigram_featuresets_train



##############################################################################


        classifier1 = NaiveBayesClassifier.train(featuresets_train)
        classifier2 = MaxentClassifier.train(featuresets_train)
        classifier3 = nltk.classify.SklearnClassifier(LinearSVC())
        classifier3.train(featuresets_train)


        refsets = collections.defaultdict(set)
        testsets1 = collections.defaultdict(set)
        testsets2 = collections.defaultdict(set)
        testsets3 = collections.defaultdict(set)

        for i, (feats, label) in enumerate(featuresets_test):
            refsets[label].add(i)
            observed1 = classifier1.classify(feats)
            observed2 = classifier2.classify(feats)
            observed3 = classifier3.classify(feats)
            testsets1[observed1].add(i)
            testsets2[observed2].add(i)
            testsets3[observed3].add(i)

        accuracy_list_nb.append(nltk.classify.accuracy(classifier1, featuresets_test))
        f_measure_list_nb.append(nltk.metrics.f_measure(refsets['not'], testsets1['not']))
        accuracy_list_svm.append(nltk.classify.accuracy(classifier3, featuresets_test))
        f_measure_list_svm.append(nltk.metrics.f_measure(refsets['not'], testsets3['not']))
        accuracy_list_maxent.append(nltk.classify.accuracy(classifier2, featuresets_test))
        f_measure_list_maxent.append(nltk.metrics.f_measure(refsets['not'], testsets2['not']))


 ################################################################################

    print "+-----------------------------------------------------------------+"
    print "\t\t\t\t\tbigram and unigram classification measurements"
    print "+-----------------------------------------------------------------+"
    print "\t\t\t\t\t\t\tmax accuracy \t number of features "
    print "Naive Bayes\t\t\t\t\t %f \t\t\t\t%d" % (max(accuracy_list_nb), (accuracy_list_nb.index(max(accuracy_list_nb))*20)+10)
    print "Maximum entropy\t\t\t\t %f \t\t\t\t%d" % (max(accuracy_list_maxent), (accuracy_list_maxent.index(max(accuracy_list_maxent))*20)+10)
    print "Support Vector Machine\t\t %f \t\t\t\t%d" % (max(accuracy_list_svm), (accuracy_list_svm.index(max(accuracy_list_svm))*20)+10)
    print "+-----------------------------------------------------------------+"
    print "+-----------------------------------------------------------------+"
    print "\t\t\t\t\t\t\tmax f-measure \t number of features "
    print "Naive Bayes\t\t\t\t\t %f \t\t\t\t%d" % (max(f_measure_list_nb), (f_measure_list_nb.index(max(f_measure_list_nb))*20)+10)
    print "Maximum entropy\t\t\t\t %f \t\t\t\t%d" % (max(f_measure_list_maxent), (f_measure_list_maxent.index(max(f_measure_list_maxent))*20)+10)
    print "Support Vector Machine\t\t %f \t\t\t\t%d" % (max(f_measure_list_svm), (f_measure_list_svm.index(max(f_measure_list_svm))+1)*20)
    print "+-----------------------------------------------------------------+"
################################################################################

    print " time taken for the classification process %f sec " % (time() - t0)
#####################################################################################################
    x_axis = [i for i in range(10, 200, 20)]
    plt.figure(facecolor='white')
    fig1, = plt.plot(x_axis, accuracy_list_nb, 'r*-', label='Naive bayes accuracy')
    fig2, = plt.plot(x_axis, f_measure_list_nb, 'ro-', label='Naive bayes f-measure')
    fig3, = plt.plot(x_axis, accuracy_list_svm, 'g*-', label='SVM accuracy')
    fig4, = plt.plot(x_axis, f_measure_list_svm, 'go-', label='SVM f-measure')
    fig5, = plt.plot(x_axis, accuracy_list_maxent, '*-', label='max Entropy accuracy')
    fig6, = plt.plot(x_axis, f_measure_list_maxent, 'o-', label='max Entropy f-measure')

    plt.xlabel('Number of features')
    plt.ylabel('Results')
    plt.title('Results of the classification using unigrams and bigrams')
    plt.legend(handles=[fig1, fig2, fig3, fig4, fig5, fig6], loc=4)
    plt.show()


t0 = time()

filename = 'all_tweets.txt'
lines = preprocess.main(filename)


bigram_evaluation(lines)
unigram_evaluation(lines)
uni_and_bi_validation(lines)
