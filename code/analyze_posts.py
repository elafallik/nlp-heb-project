from data_preparation import get_topic_pred_by_posts
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def show_topics_by_party(posts):
    parties = posts['party'].values
    topic_pred = posts['pred_topic_tags'].values
    parties_names, idx2 = np.unique(parties, return_index=True)
    parties_topics = {}
    for party in parties_names:
        parties_topics[party] = []

    for i in range(len(parties)):
        party = parties[i]
        parties_topics[party].append(topic_pred[i])

    for party in parties_names:
        parties_topics[party] = np.concatenate(parties_topics[party])

    topics = list(parties_topics.values())

    res = []
    for l in topics:
        (unique, counts) = np.unique(l, return_counts=True)
        temp = np.zeros(4, dtype=int)
        unique = np.array(unique, dtype=int)
        temp[unique] = counts
        res.append(temp)

    def invert_words(words):
        return [w[::-1] for w in words]

    res = np.array(res)
    res = (res / np.sum(res, axis=1)[:, np.newaxis]).transpose()
    temp = {}
    temp['party'] = invert_words(parties_names) * 3
    tags = invert_words(['טראמפ'] * len(parties_names) + ['נתניהו'] * len(parties_names) + ['חוק הלאום'] * len(parties_names))
    temp['tag'] = tags
    temp['percentage'] = np.reshape(res[1:], (len(res[0]) * 3))
    new_df = pd.DataFrame(temp, columns=pd.Index(['party', 'tag', 'percentage']))

    fig, ax = plt.subplots(figsize=(12,5))
    sns.barplot(x='party', y='percentage', hue='tag', data=new_df)
    plt.title('Topic percentage by party')
    plt.savefig('plots/topic_percentage_by_party.png')
    plt.show()


def show_sentiment_by_member(posts, party_name, topic_tag):
    parties = posts['party'].values
    idx = parties == party_name
    posts = posts[idx]
    sentiment_pred = posts[topic_tag].values
    idx = np.array([len(p) != 0 for p in sentiment_pred])
    posts = posts[idx]
    sentiment_pred = posts[topic_tag].values
    members = posts['author'].values
    members_name, idx2 = np.unique(members, return_index=True)
    members_sentiment = {}
    for m in members_name:
        members_sentiment[m] = []

    for i in range(len(members)):
        members_sentiment[members[i]].append(sentiment_pred[i])

    for m in members_name:
        members_sentiment[m] = np.concatenate(members_sentiment[m])

    sentiments = list(members_sentiment.values())

    res = []
    for l in sentiments:
        (unique, counts) = np.unique(l, return_counts=True)
        unique += 1
        temp = np.zeros(3, dtype=int)
        unique = np.array(unique, dtype=int)
        temp[unique] = counts
        res.append(temp)

    def invert_words(words):
        return [w[::-1] for w in words]

    res = np.array(res)
    res = (res / np.sum(res, axis=1)[:, np.newaxis]).transpose()
    temp = {}
    temp['member'] = invert_words(members_name) * 3
    tags = ['negative'] * len(members_name) + ['neutral'] * len(members_name) + ['positive'] * len(members_name)
    temp['tag'] = tags
    temp['percentage'] = np.reshape(res, (len(res[0]) * 3))
    new_df = pd.DataFrame(temp, columns=pd.Index(['member', 'tag', 'percentage']))

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x='member', y='percentage', hue='tag', data=new_df)
    plt.title('Sentiment percentage by member in ' + invert_words([party_name])[0] + ' party for topic ' + topic_tag)
    plt.savefig('plots/sentiment_percentage_by_member_licud.png')
    plt.show()


if __name__ == '__main__':
    posts = get_topic_pred_by_posts("final_tagged/posts_list_3_pred_sentiment")
    show_sentiment_by_member(posts, 'הליכוד', '3')