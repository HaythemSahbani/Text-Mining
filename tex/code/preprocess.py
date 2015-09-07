from nltk import RegexpTokenizer
import contractions
import stop_words
import re
from nltk.stem.porter import PorterStemmer


class Preprocess:

    def __init__(self):
        pass

    @staticmethod
    def remove_useless_elm(list_):
        """
        Removes links and rt from a token list
        :param list_:
        :return:
        """
        # removes the URL and the retweet 'rt' tokens
        delet_index = [index for index, token in enumerate(list_)
                       if token.startswith("http")
                       or token.startswith("www")
                       or token.lower().startswith("rt")]
        delet_index.reverse()
        for _ in delet_index:
            list_.pop(_)
        return list_

    @staticmethod
    def remove_stopwords(tokens):
        """
        remove stopwords (i.e., "scaffold words" in English w/o much meaning)
        """
        return [token for token in tokens if token not in stop_words.stopwords_list]

    def expand_contraction(self, text):
        """
        expand contractions in a list of tokens
        """
        text = re.sub(r"(\t)", " ", text)  # remove tabulation
        tokens = self.remove_useless_elm(text.split(" "))
        # tokens = re.sub(r"(\t)", " ", " ".join(tokens))  # remove tabulation
        tokens = re.sub("([!,;:?\.]*)", '', " ".join(tokens))  # pattern: remove dots.
        tokens = tokens.lower().split(" ")
        result_tokens = []
        for token in tokens:
            is_contraction = False
            for contraction, expansion in contractions.contractions_dict:
                if token == contraction or token == contraction.replace("'", ""):
                    result_tokens += expansion.split()
                    is_contraction = True
                    break

            if not is_contraction:
                result_tokens.append(token)

        return " ".join(result_tokens)


class TweetTokenizer:
    pattern = r'''  (?x) # set flag to allow verbose regexps
                    (\@\w*)+  # takes the names but also emails
                    |(\#\w*)+  # takes the htags
                    # | (\http://\S*) # takes http links
                    # | (\www.\S*) # takes www links
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

    def tokenize(self, text):
        """
        :param tweet_list:
        :type list:
        :return: tokens
        This tokenizer uses the nltk RegexpTokenizer.
        """
        tokenizer = RegexpTokenizer(self.pattern)
        tokens = tokenizer.tokenize(text)
        return tokens

    @staticmethod
    def stem(tokens):
        stems = [PorterStemmer().stem(token) for token in tokens]
        return stems


def main(filename):

    # open the file containing the dataset
    f = open(filename)
    lines = f.readlines()
    f.close()

    # preprocess the tweets
    for line in lines:
        ll = TweetTokenizer().tokenize(Preprocess().expand_contraction(line))
        lines[lines.index(line)] = ll[0], Preprocess().remove_stopwords(ll[1:])

    #  delete the tweets that have 1 word or less after preprocessing.
    index_list = [index for index, elem in enumerate(lines) if len(elem[1]) < 2]
    index_list.reverse()
    for _ in index_list:
        lines.pop(_)

    return lines
if __name__ == '__main__':
    filename = "all_tweets.txt"

    print main(filename=filename)[:10]