

class HashtagClassifier:
    def __init__(self):
        pass

    @staticmethod
    def hashtag_classifier(tweet_list):
        """
        :param list containing json dictionary:
        :return:
        """
        topic_dict = dict([])
        topic_dict["no_hashtag_tweet"] = []
        for tweet in tweet_list:
            no_hashtag_test = True
            for word in tweet[1]:
                if word.startswith("#"):
                    no_hashtag_test = False
                    try:
                        topic_dict[word].append(tweet[0])
                    except:
                        topic_dict[word] = [tweet[0]]
            if no_hashtag_test:
                topic_dict["no_hashtag_tweet"].append(tweet)
        return topic_dict

    @staticmethod
    def get_most_frequent_hashtag(dic):
        lst = [word for word, frequency in sorted(dic.items(), key=lambda t: len(t[1]), reverse=True)]
        print("Most frequent tweets:", lst[1:6])
        return lst




def main():
    import preprocess

    filename = "all_tweets.txt"
    lines = preprocess.main(filename)

    dic = HashtagClassifier().hashtag_classifier(lines)

    print len(dic['no_hashtag_tweet'])
    i = 0
    for key in dic:
        if len(dic[key]) > 9 and key != "no_hashtag_tweet":
            print key + "\t", dic[key]
            # print len(dic[key])
            i += 1
    print i

if __name__ == "__main__":
    main()


