import pandas as pd
import numpy as np
import os
SOURCE_DIR = os.getcwd() + "/"


def create_sentences_list(posts_list, sentences_list_name):
    data = pd.read_json(SOURCE_DIR + "data/" + posts_list + ".json", encoding="utf-8")
    all_topic_tags = np.array(sum(data['topic_tags'].values, []))
    all_sentiment_tags = np.array(sum(data['sentiment_tags'].values, []))
    all_sentences = np.array(sum(data['text'].values, []))
    lengths = np.array([len(post) for post in data['text'].values])
    post_num = np.concatenate([[i] * lengths[i] for i in range(len(lengths))])
    post_idx = np.concatenate([range(l) for l in lengths])
    temp = {'post_num': post_num,
            'post_idx': post_idx,
            'text': all_sentences,
            'topic_tag': all_topic_tags,
            'sentiment_tag': all_sentiment_tags}
    new_df = pd.DataFrame(temp, columns=['post_num', 'post_idx', 'text', 'topic_tag', 'sentiment_tag'])
    new_df.to_csv(SOURCE_DIR + "data/" + sentences_list_name + ".csv", index=False, encoding="utf-8")
    new_df.to_json(SOURCE_DIR + "data/" + sentences_list_name + ".json")


def split_train_test(data_file, percentage=30):
    data = pd.read_json(SOURCE_DIR + "data/" + data_file + ".json", encoding="utf-8")
    data = data.sample(frac=1).reset_index(drop=True)  # shuffle rows
    test_size = int(len(data) * (percentage / 100))
    train = data[:len(data) - test_size]
    test = data[len(data) - test_size:]
    train.to_csv(SOURCE_DIR + "data/" + data_file + "_train.csv", index=False, encoding="utf-8")
    train.to_json(SOURCE_DIR + "data/" + data_file + "_train.json")
    test.to_csv(SOURCE_DIR + "data/" + data_file + "_test.csv", index=False, encoding="utf-8")
    test.to_json(SOURCE_DIR + "data/" + data_file + "_test.json")


def get_sentences(data_file, return_topic_tags=False, return_sentiment_tags=False):
    df = pd.read_json(SOURCE_DIR + "data/" + data_file + ".json", encoding="utf-8")
    sentences = df['text'].values
    if return_topic_tags and return_sentiment_tags:
        topic_tags = df['topic_tag'].values
        sentiment_tags = df['sentiment_tag'].values
        return sentences, topic_tags, sentiment_tags
    elif return_topic_tags:
        topic_tags = df['topic_tag'].values
        return sentences, topic_tags
    elif return_sentiment_tags:
        sentiment_tags = df['sentiment_tag'].values
        return sentences, sentiment_tags
    return sentences


def get_topic_tags(data_file):
    df = pd.read_json(SOURCE_DIR + "data/" + data_file + ".json", encoding="utf-8")
    topic_tags = df['topic_tag'].values
    return topic_tags


def get_sentiment_tags(data_file):
    df = pd.read_json(SOURCE_DIR + "data/" + data_file + ".json", encoding="utf-8")
    sentiment_tags = df['sentiment_tag'].values
    return sentiment_tags


def save_topic_pred(sentences_file, pred_topic_tags):
    data = pd.read_json(SOURCE_DIR + "data/" + sentences_file + ".json", encoding="utf-8")
    temp = {}
    for i in data.columns:
        temp[i] = data[i]
    temp['pred_topic_tag'] = pred_topic_tags
    new_df = pd.DataFrame(temp, columns=pd.Index(list(data.columns) + ['pred_topic_tag']))
    new_df.to_csv(SOURCE_DIR + "out/" + sentences_file + "_pred_topics.csv", index=False, encoding="utf-8")
    new_df.to_json(SOURCE_DIR + "out/" + sentences_file + "_pred_topics.json")


def save_sentiment_pred(sentences_file, pred_sentiment_tags):
    data = pd.read_json(SOURCE_DIR + "data/" + sentences_file + ".json", encoding="utf-8")
    temp = {}
    for i in data.columns:
        temp[i] = data[i]
    temp['pred_sentiment_tag'] = pred_sentiment_tags
    new_df = pd.DataFrame(temp, columns=pd.Index(list(data.columns) + ['pred_sentiment_tag']))
    new_df.to_csv(SOURCE_DIR + "out/" + sentences_file + "_pred_sentiment.csv", index=False, encoding="utf-8")
    new_df.to_json(SOURCE_DIR + "out/" + sentences_file + "_pred_sentiment.json")


def get_sentences_pred(data_file, return_topic_pred=False, return_sentiment_pred=False):
    df = pd.read_json(SOURCE_DIR + "out/" + data_file + ".json", encoding="utf-8")
    sentences = df['text'].values
    if return_topic_pred and return_sentiment_pred:
        topic_tags = df['pred_topic_tag'].values
        sentiment_tags = df['pred_sentiment_tag'].values
        return sentences, topic_tags, sentiment_tags
    elif return_topic_pred:
        topic_tags = df['pred_topic_tag'].values
        return sentences, topic_tags
    elif return_sentiment_pred:
        sentiment_tags = df['pred_sentiment_tag'].values
        return sentences, sentiment_tags
    return sentences


def get_pred_sentiment_tags(data_file):
    df = pd.read_json(SOURCE_DIR + "out/" + data_file + ".json", encoding="utf-8")
    sentiment_tags = df['pred_sentiment_tags'].values
    return sentiment_tags


def get_pred_topic_tags(data_file):
    df = pd.read_json(SOURCE_DIR + "out/" + data_file + ".json", encoding="utf-8")
    topic_tags = df['pred_topic_tag'].values
    return topic_tags


def sort_sentences_list(sentences_list):
    data = pd.read_json(SOURCE_DIR + "out/" + sentences_list + ".json", encoding="utf-8")
    post_num = data['post_num'].values
    idx = np.argsort(post_num)
    temp = {}
    for i in data.columns:
        temp[i] = np.array(data[i])[idx]
    new_df = pd.DataFrame(temp, columns=data.columns)

    _, idx2 = np.unique(new_df['post_num'].values, return_index=True)
    idx_new = np.concatenate([np.argsort(new_df['post_idx'].values[idx2[i]: idx2[i + 1]]) + idx2[i] for i in range(len(idx2) - 1)])
    temp = {}
    for i in new_df.columns:
        temp[i] = np.array(new_df[i])[idx_new]
    new_df = pd.DataFrame(temp, columns=new_df.columns)
    new_df.to_csv(SOURCE_DIR + "out/" + sentences_list + "_sorted.csv", index=False, encoding="utf-8")
    new_df.to_json(SOURCE_DIR + "out/" + sentences_list + "_sorted.json")


def save_topic_pred_by_post(pred_file, post_file):
    data = pd.read_json(SOURCE_DIR + "out/" + pred_file + ".json", encoding="utf-8")
    posts = pd.read_json(SOURCE_DIR + "data/" + post_file + ".json", encoding="utf-8")
    post_num = data['post_num'].values
    post_idx = data['post_idx'].values
    pred_tags = data['pred_topic_tag'].values

    pred_topic_tags = [[] for i in range(len(posts))]

    for i, j in zip(post_num, range(len(post_idx))):
        pred_topic_tags[i].append(pred_tags[j])

    temp = {}
    for i in posts.columns:
        temp[i] = posts[i]
    temp['pred_topic_tags'] = pred_topic_tags
    new_df = pd.DataFrame(temp, columns=pd.Index(list(posts.columns) + ['pred_topic_tags']))
    new_df.to_csv(SOURCE_DIR + "out/" + post_file + "_pred_topics.csv", index=False, encoding="utf-8")
    new_df.to_json(SOURCE_DIR + "out/" + post_file + "_pred_topics.json")


def get_topic_pred_by_posts(pred_posts_file):
    posts = pd.read_json(SOURCE_DIR + "out/" + pred_posts_file + ".json", encoding="utf-8")
    return posts


def save_sentiment_pred_by_post(pred_file, post_file, tag_name):
    data = pd.read_json(SOURCE_DIR + "out/" + pred_file + ".json", encoding="utf-8")
    posts = pd.read_json(SOURCE_DIR + "data/" + post_file + ".json", encoding="utf-8")
    post_num = data['post_num'].values
    post_idx = data['post_idx'].values
    pred_tags = data['pred_sentiment_tag'].values

    pred_sentiment_tags = [[] for i in range(len(posts))]

    for i, j in zip(post_num, range(len(post_idx))):
        pred_sentiment_tags[i].append(pred_tags[j])

    temp = {}
    for i in posts.columns:
        temp[i] = posts[i]
    temp[tag_name] = pred_sentiment_tags
    new_df = pd.DataFrame(temp, columns=pd.Index(list(posts.columns) + [tag_name]))
    new_df.to_csv(SOURCE_DIR + "out/" + post_file + "_" + tag_name + "_pred_sentiment.csv", index=False, encoding="utf-8")
    new_df.to_json(SOURCE_DIR + "out/" + post_file + "_" + tag_name + "_pred_sentiment.json")



if __name__ == '__main__':
    # split_train_test('final_tagged/sentences_list_shuffled_tagged_topics_3', 30)

    create_sentences_list('final_tagged/posts_list',
                          'final_tagged/sentences_list')
    sort_sentences_list('final_tagged/sentences_list')
