

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
        print("Most frequent tweets:", lst[:6])
        return lst




def main():
    import preprocess

    filename = "all_tweets.txt"
    lines = preprocess.main(filename)

    dic = NameClassifier().name_classifier(lines)

    print len(dic)
    s = NameClassifier().get_most_frequent_names(dic)
    i = 0
    for key in s:
        # Only usernames that occur in more than 4 tweets are taken in consideration
        if len(dic[key]) > 4 and key != "no_name_tweet":
            i += 1
            print key + "\t", dic[key]
    print i


    #for key in dic:
    #    if len(dic[key]) > 2:
    #        print key + "\t", dic[key]
    # print dic

if __name__ == "__main__":
    main()


