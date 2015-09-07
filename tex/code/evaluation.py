__author__ = 'Haythem Sahbani'
import preprocess
from feature_extraction import FeatureExtraction
from hashtag_classifier import HashtagClassifier
from nltk import NaiveBayesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import nltk
import collections
import nltk.metrics
from sklearn.svm import LinearSVC


filename = "all_tweets.txt"

ftr = FeatureExtraction(150)

lines = preprocess.main(filename=filename)

all_tweets = " ".join([" ".join(line[1]) for line in lines])
ftr.most_frequent_bigrams(all_tweets)

unigram_featuresets = [(ftr.unigram_features(line[1]), line[0]) for line in lines]

random.shuffle(unigram_featuresets)

train_set, test_set = unigram_featuresets[:2000], unigram_featuresets[2001:]

classifier = NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features()
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)

    testsets[observed].add(i)

print 'pos precision:', nltk.metrics.precision(refsets['polit'], testsets['polit'])
print 'pos recall:', nltk.metrics.recall(refsets['polit'], testsets['polit'])
print 'pos F-measure:', nltk.metrics.f_measure(refsets['polit'], testsets['polit'])
print 'neg precision:', nltk.metrics.precision(refsets['not'], testsets['not'])
print 'neg recall:', nltk.metrics.recall(refsets['not'], testsets['not'])
print 'neg F-measure:', nltk.metrics.f_measure(refsets['not'], testsets['not'])

classifier1 = nltk.classify.SklearnClassifier(LinearSVC())
classifier1.train(train_set)
print "svm class ", nltk.classify.accuracy(classifier1, test_set)