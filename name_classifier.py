

class NameClassifier:
    def __init__(self):
        pass

    @staticmethod
    def name_classifier(tweet_list):
        """
        :param list containing json dictionary:
        :return:
        """
        topic_dict = dict([])
        topic_dict["no_name_tweet"] = []
        for tweet in tweet_list:
            no_hashtag_test = True
            for word in tweet[1]:
                if word.startswith("@") and len(word) > 1:
                    no_hashtag_test = False
                    try:
                        topic_dict[word].append(tweet[0])
                    except:
                        topic_dict[word] = [tweet[0]]
            if no_hashtag_test:
                topic_dict["no_name_tweet"].append(tweet)
        return topic_dict

    @staticmethod
    def get_most_frequent_names(dic):
        lst = [word for word, frequency in sorted(dic.items(), key=lambda t: len(t[1]), reverse=True)]
        lst.pop(0)  # deleting the no_name_tweet entry

        return lst[:10]




def main():
    import preprocess

    filename = "all_tweets.txt"
    lines = preprocess.main(filename)

    dic = NameClassifier().name_classifier(lines)

    print 'The 10 most frequent usernames: ', NameClassifier().get_most_frequent_names(dic)
    print "number of tweets without a user name is %d, it's %f  of the data set" % (len(dic['no_name_tweet']), 100*len(dic['no_name_tweet'])/len(lines))

if __name__ == "__main__":
    main()


