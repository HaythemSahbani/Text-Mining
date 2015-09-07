__author__ = 'Haythem Sahbani'

######################################
#
# This file contains
# the feature extraction classes
#
#
#######################################

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import bigrams, FreqDist


class FeatureExtraction():
    def __init__(self, n_features=60):
        """
        :param n_features: number of features used
        :return:
        """
        self.n_features = n_features

    def set_unigram_features_list(self, token):
        """
        a seter for the unigram_features_list
        :param token:  string like
        :return:
        """
        self.unigram_features_list.append(token)

    def most_frequent_unigrams(self, text):
        """
        Gives the most frequent unigram features, the number is defined by: n_features
        :param text: plain text
        :return:  list of unigram features
        """
        self.unigram_features_list = [word for (word, freq) in nltk.FreqDist(text.split(" ")).most_common(self.n_features)]
        return self.unigram_features_list

    def tf_idf_features(self, tweets, n_grams=1):
        """
        Extracts tf-idf based features using the scikit learn TfidfVectorizer.
        :param tweets: List of tweets
        :param n_grams: takes 1 (unigram) or 2 (bigram) as values.
        :return: tf idf based bag of words.
        """
        tfidf_vect = TfidfVectorizer(max_df=0.95, max_features=self.n_features,
                                     min_df=2, stop_words=None, ngram_range=(n_grams, n_grams))

        tfidf_vect.fit_transform(tweets)

        if n_grams == 1:
            self.unigram_features_list = tfidf_vect.get_feature_names()
        else:
            self.bigram_features_list = tfidf_vect.get_feature_names()

        return tfidf_vect.get_feature_names()

    def unigram_features(self, text):
        """
        Takes a plain text as parameters and returns a dictionary
        suited as the classifiers input.
        :param text:
        :return:
        """
        text_words = set(text)
        features = {}
        for word in self.unigram_features_list:
            features['contains(%s)' % word] = (word in text_words)
        return features

    def most_frequent_bigrams(self, text):
        """
        Gives the most frequent bigram features, the number is defined by: n_features
        :param text: plain text
        :return: list of bigram features
        """
        self.bigram_features_list = [bigram for bigram, freq in FreqDist(bigrams(text.split(" "))).most_common(self.n_features)]
        return self.bigram_features_list

    def bigram_features(self, tokens):
        """
        Takes list of tokens as parameters and returns a bigram dictionary
        :param tokens:
        :param :
        :return:
        """
        bigram_list = [" ".join(word) for word in bigrams(tokens)]
        features = {}
        for word in self.bigram_features_list:
            features['contains(%s)' % " ".join(word)] = (" ".join(word) in bigram_list)
        return features


class PatternsFeatures():

    def __init__(self):
        pass

    @staticmethod
    def pattern_classifier(tweet_list, pattern):
        """
        :param tweet_list: A list containing tweets.
        :param pattern: The special pattern looking for, eather @ or #
        :return: dictionary containing the word pattern as key and it's label as values.
        """
        topic_dict = dict([])
        topic_dict["no_pattern_tweet"] = []
        for tweet in tweet_list:
            no_pattern_test = True
            for word in tweet[1]:
                if word.startswith(pattern):
                    no_pattern_test = False
                    try:
                        topic_dict[word].append(tweet[0])
                    except:
                        topic_dict[word] = [tweet[0]]
            if no_pattern_test:
                topic_dict["no_pattern_tweet"].append(tweet)
        return topic_dict

    @staticmethod
    def get_most_frequent_pattern(dic, n=10):
        """
        :param dic: dictionary, the output of pattern_classifier(tweet_list, pattern)
        :param n: number of feature extracted
        :return: list containing the n most used keys in dic
        """
        lst = [word for word, frequency in sorted(dic.items(), key=lambda t: len(t[1]), reverse=True)]
        lst.pop(0)  # contains the "no_pattern_tweet" entry.
        return lst[:n]



def main():
    """
    Test for the feature extraction class
    :return:
    """
    import preprocess
    ftr = FeatureExtraction(6)
    filename = "all_tweets.txt"
    lines = preprocess.main(filename)

    all_tweets = " ".join([" ".join(line[1]) for line in lines])

    print "The most frequent bigrams are :", ftr.most_frequent_bigrams(all_tweets)
    print "The most frequent unigrams are :", ftr.most_frequent_unigrams(all_tweets)

    hashtag_dic = PatternsFeatures().pattern_classifier(lines, '#')

    print 'The 10 most frequent hashtags', PatternsFeatures().get_most_frequent_pattern(hashtag_dic)
    print "number of tweets without hashtag is %d, it's %f percent of the data set" % (len(hashtag_dic['no_pattern_tweet']), 100*len(hashtag_dic['no_pattern_tweet'])/len(lines))

    name_dic = PatternsFeatures().pattern_classifier(lines, '@')

    print 'The 10 most frequent usernames: ', PatternsFeatures().get_most_frequent_pattern(name_dic)
    print "number of tweets without a user name is %d, it's %f  of the data set" % (len(name_dic['no_pattern_tweet']), 100*len(name_dic['no_pattern_tweet'])/len(lines))

if __name__ == "__main__":
        main()