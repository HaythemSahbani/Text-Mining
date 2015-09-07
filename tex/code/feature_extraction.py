__author__ = 'Haythem Sahbani'
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import bigrams, FreqDist


class FeatureExtraction():
    def __init__(self, n_features=200):
        """

        :param n_features: number of features
        :return:
        """
        self.n_features = n_features


    def most_frequent_unigrams(self, text):
        """
        :param text: plain text
        :return:  n most frequent words in the token list
        """
        self.unigram_features_list = [word for (word, freq) in nltk.FreqDist(text.split(" ")).most_common(self.n_features)]
        return self.unigram_features_list

    def tf_idf_features(self, tweets):
        """

        :param tweets: List of tweets
        :return: tf idf based words
        """
        tfidf_vect = TfidfVectorizer(max_df=0.95, max_features=self.n_features, min_df=2, stop_words=None)
        tfidf_vect.fit_transform(tweets)
        self.unigram_features_list = tfidf_vect.get_feature_names()

    def unigram_features(self, text):  # [_document-classify-extractor]
        """
        Takes a plain text as parameters and returns a dictionary
        suited for the nltk Naive Bayes classifier
        :param text:
        :param feature_set:
        :return:
        """
        text_words = set(text) # [_document-classify-set]
        features = {}
        for word in self.unigram_features_list:
            features['contains(%s)' % word] = (word in text_words)
        return features

    def most_frequent_bigrams(self, text):
        """
        :param text: plain text
        :return: n most frequent bigrams
        """
        self.bigram_features_list = [bigram for bigram, freq in FreqDist(bigrams(text.split(" "))).most_common(self.n_features)]
        return self.bigram_features_list

    def bigram_features(self, text):  # [_document-classify-extractor]
        """
        Takes a plain text as parameters and returns a bigram dictionary
        :param text:
        :param :
        :return:
        """
        # text_words = set(text) # [_document-classify-set]
        features = {}
        for word in self.bigram_features_list:
            features['contains(%s)' % " ".join(word)] = (" ".join(word) in bigrams(text))
        return features


def main():
    """
    Test for the feature extraction class
    :return:
    """
    import preprocess

    ftr = FeatureExtraction(3)
    filename = "all_tweets.txt"

    lines = preprocess.main(filename)

    all_tweets = " ".join([" ".join(line[1]) for line in lines])


    print ftr.most_frequent_bigrams(all_tweets)
    print ftr.most_frequent_unigrams(all_tweets)

    bigram_featuresets = [(ftr.bigram_features(line[1]), line[0]) for line in lines]
    unigram_featuresets = [(ftr.unigram_features(line[1]), line[0]) for line in lines]
    # print bigram_featuresets
    # print unigram_featuresets


if __name__ == "__main__":
        main()