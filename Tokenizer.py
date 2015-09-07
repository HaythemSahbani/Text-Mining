__author__ = 'Haythem Sahbani'

import nltk
import operator
from nltk.tokenize import RegexpTokenizer
import numpy as np

class TweetTokenizer:
    pattern = r'''  (?x) # set flag to allow verbose regexps
                    (\@\w*)+  # takes the names but also emails
                    #(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)
                    |(\#\w*)+  # takes the htags
                    | (\http://\S*) # takes http links
                    | (\www.\S*) # takes www links
                    |([A-Z]\.)+ # abbreviations, e.g. U.S.A.
                    | \w+(-\w+)* # words with optional internal hyphens
                    | \$?\d+(\.\d+)?%? # currency and percentages, e.g. $12.40, 82%

                    | ^(\.+) # ellipsis  e.g. .....
                    | (%[][.,;"'?():-_`])+ # these are separate tokens
                    #| (?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+) doesn't take mail address
                '''

    def __init__(self, pattern=pattern):
        self.pattern = pattern
        pass


    def tokenize(self, tweet_list):
        """
        :param tweet_list:
        :type list:
        :return: tokens
        This tokenizer uses the nltk RegexpTokenizer.
        """
        tokenizer = RegexpTokenizer(self.pattern)
        tokens = tokenizer.tokenize(tweet_list)

        for item in tokens:
            # print("lower: ", tokens.index(item), item)
            tokens[tokens.index(item)] = item.lower()

            if item.startswith("http://") or item.startswith("www."):
                # print("add <URL> ", tokens.index(item))
                tokens[tokens.index(item)] = "<URL>"

        for item in tokens:
                if item == "rt":
                    # print("remove rt ", tokens.index(item))
                    tokens.pop(tokens.index(item))
        return tokens

    @staticmethod
    def stem():
        pass


def nltk_dictionary(text_file, result_file):
    """  This function creates a dictionary from a text file.
        It uses the nltk package
        It saves the result in a second text file
    """
    # Open the result file and make it writable.
    result = open(result_file, 'w')

    # Open and read the text file.
    text = open(text_file, 'r').read()
    # Tokenize the text and get rid of the symbols, this feature uses RegexpTokenizer.
    # 1- Create the tokenizer object.
    tokenizer = RegexpTokenizer(r'\w+')
    # 2- Tokenize the text and eliminate the symbols.
    text = tokenizer.tokenize(text.lower())
    # Create the dictionary using the FredDist function.
    dictionary = nltk.FreqDist(text)
    # This loop sorts the result by the frequency of the word then stores it in the result file.
    for word, frequency in sorted(dictionary.items(), key=operator.itemgetter(1)):
        result.write(word + "\t" + str(frequency) + "\n")

    # Close the result file to free the RAM.
    result.close()

