import pandas as pd
import numpy as np
import os
SOURCE_DIR = os.getcwd() + "/"


def tag_topics(data_file, num_topics, tag_commands):
    num_topics += 2  # one because we start from 1 and one because of "other topic"
    try:
        data = pd.read_json(SOURCE_DIR + "data/" + data_file + "_tagged_topics.json", encoding="utf-8")
        tags = data["topic_tags"].values
    except:
        data = pd.read_json(SOURCE_DIR + "data/" + data_file + ".json", encoding="utf-8")
        tags = [[-1]] * len(data)

    posts = data["text"].values
    print(data_file)
    print(len(posts))

    finish = 0
    i = np.random.randint(0, len(data))
    while not finish:
        if tags[i][0] == -1:
            print(i)
            post = posts[i]
            print(post)
            try:
                skip = int(input("skip = 1\n"))
            except:
                skip = 1
                print('incorrect input')
            if skip == 1:
                tag = [-1]
            else:
                tag = []
                for s in post:
                    print(s)
                    t = -1
                    while len(np.where(np.arange(0, num_topics) == t)[0]) == 0:
                        try:
                            t = int(input("tag: else=1, " + tag_commands + "\n"))
                        except ValueError:
                            print("error, try again")
                            t = -1
                    tag.append(t - 1)
            tags[i] = tag
            finish = -1
            while finish != 0 and finish != 1 and finish != 2:
                try:
                    finish = int(input("continue? 1 continue, 2 finish, 3 tag last again"))
                except ValueError:
                    print("error, try again")
                    finish = -1
                finish -= 1
        if finish != 2:
            i = np.random.randint(0, len(data))
        else:
            tags[i] = [-1]
            finish = 0

    data["topic_tags"] = tags
    data.to_csv(SOURCE_DIR + "data/" + data_file + "_tagged_topics.csv", index=False, encoding="utf-8")
    data.to_json(SOURCE_DIR + "data/" + data_file + "_tagged_topics.json")


def tag_sentiment(data_file):
    try:
        data = pd.read_json(SOURCE_DIR + "data/" + data_file + "_tagged_sentiment.json", encoding="utf-8")
        sentiment_tags = data["sentiment_tags"].values
    except:
        data = pd.read_json(SOURCE_DIR + "data/" + data_file + ".json", encoding="utf-8")
        sentiment_tags = [[-2]] * len(data)

    posts = data["text"].values
    topics_tags = data["topic_tags"].values
    print(data_file)
    print(len(posts))

    finish = 0
    i = 0
    while not finish and i != len(data):
        if sentiment_tags[i] is None or sentiment_tags[i][0] == -2:
            topics_tag = topics_tags[i]
            print(i)
            post = posts[i]
            print(post)
            tag = []
            print(topics_tag)
            for j in range(len(post)):
                s = post[j]
                if topics_tag[j] == 0:
                    tag.append(2)
                else:
                    print(s)
                    t = -2
                    while t != -1 and t != 0 and t != 1:
                        try:
                            t = int(input("tag: 1 negative, 2 neutral, 3 positive\n"))
                        except ValueError:
                            print("error, try again")
                            t = -2
                        t -= 2
                    tag.append(t)
            sentiment_tags[i] = tag
            finish = -1
            while finish != 0 and finish != 1 and finish != 2:
                try:
                    finish = int(input("continue? 1 continue, 2 finish, 3 tag last again"))
                except ValueError:
                    print("error, try again")
                    finish = -1
                finish -= 1
        if finish != 2:
            i += 1
        else:
            sentiment_tags[i] = [-2]
            finish = 0

    data["sentiment_tags"] = sentiment_tags
    data.to_csv(SOURCE_DIR + "data/" + data_file + "_tagged_sentiment.csv", index=False, encoding="utf-8")
    data.to_json(SOURCE_DIR + "data/" + data_file + "_tagged_sentiment.json")


def save_topics_tagged(data_file):
    data = pd.read_json(SOURCE_DIR + "data/" + data_file + "_tagged_topics.json", encoding="utf-8")
    tags = data["topic_tags"].values
    tagged_posts = data[np.array([s[0] for s in tags]) != -1]
    tagged_posts.to_csv(SOURCE_DIR + "data/" + data_file + "_only_tagged_topics.csv", index=False, encoding="utf-8")
    tagged_posts.to_json(SOURCE_DIR + "data/" + data_file + "_only_tagged_topics.json")


def save_topic_sentences_by_tag(sentences_file, topic_tags):
    data = pd.read_json(SOURCE_DIR + "data/" + sentences_file + ".json", encoding="utf-8")
    tags = data["topic_tag"].values
    for i in topic_tags:
        tagged_sentences = data[tags == i]
        tagged_sentences.to_csv(SOURCE_DIR + "data/" + sentences_file + "_tagged_topics_" + str(i) + ".csv", index=False, encoding="utf-8")
        tagged_sentences.to_json(SOURCE_DIR + "data/" + sentences_file + "_tagged_topics_" + str(i) + ".json")


if __name__ == '__main__':
    # tag_topics('hokhaleom/hokhaleom_morph300_400', 4, 'trump=2, netanyahu=3, hokhaleom=4')
    save_topics_tagged('sentiment_model/trump_morph_only_tagged_topics_tagged_sentiment')

    # tag_topics('trump/trump_morph500_600', 4, 'trump=2, netanyahu=3, hokhaleom=4')
    # save_topics_tagged('trump/trump_morph500_600')

    # tag_topics('netanyahu/netanyahu_morph700_800', 4, 'trump=2, netanyahu=3, hokhaleom=4')
    # save_topics_tagged('netanyahu/netanyahu_morph700_800')
    # tag_sentiment('final_tagged/hokhaleom_morph_only_tagged_topics')
    # save_topic_sentences_by_tag('final_tagged/sentences_list_shuffled', [0, 1, 2, 3])