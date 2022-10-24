from data_preparation import get_sentences
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from scipy import linalg
from sklearn import decomposition
import matplotlib.pyplot as plt
import fbpca
import gensim
from itertools import permutations
import os


class TopicModel:
    """
    Get data and predict k most relevant topics and topic distribution for each topic.
        data: list of sentences splits to morphemes.

    predict_topics returns:
    - a list of k most relevant topics,
        - each topic defined by list of p words and the weight of each word to the topic.
        - each topic relevance is also returned, defined by
            - for SVD models: matrix s.
            - for NMF decomposition models and LDA model: weights sum for top p words.
            - for LDA model: coherence score.  # todo couldn't get it, at the moment same as NMF models

    predict_docs_dist returns:
    - the topic distribution for each doc in the training batches.
    - predicted clusters
    """

    def __init__(self, num_top_words=10, num_top_docs=100, extra_topic=True):
        self.num_topics = 0
        self.num_top_words = num_top_words
        self.num_top_docs = num_top_docs
        self.top_topics = None
        self.top_topics_relevance = None
        self.docs_topic_dist = None
        self.extra_topic = False

    def fit(self, data, num_topics):
        self.num_topics = num_topics

    def predict_topics(self):
        return self.top_topics, self.top_topics_relevance

    def predict_docs_dist(self, certainty_prec=30):
        if self.extra_topic:
            certainty_level = np.sum(self.docs_topic_dist, axis=1)
            threshold = np.percentile(certainty_level, certainty_prec)
            extra_topic_idx = np.argwhere(certainty_level < threshold)
            extra_topic_prob = np.zeros(len(certainty_level))
            extra_topic_prob[extra_topic_idx] = 1 - certainty_level[extra_topic_idx]
            docs_topic_dist = np.append(self.docs_topic_dist, np.array([extra_topic_prob]).transpose(), axis=1)
            docs_topic_dist = docs_topic_dist / np.linalg.norm(docs_topic_dist, axis=1)[:, np.newaxis]
            return docs_topic_dist, np.argmax(docs_topic_dist, axis=1)
        else:
            return self.docs_topic_dist, np.argmax(self.docs_topic_dist, axis=1)

    def get_top_words(self, Vh_top, vocab):
        top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-self.num_top_words - 1:-1]]
        top_values = lambda t: np.sort(t)[:-self.num_top_words - 1:-1]
        topic_words = [(top_words(t), top_values(t)) for t in Vh_top]
        return np.transpose(np.array(topic_words), (0, 2, 1))


class MatrixTopicModel(TopicModel):

    def __init__(self, num_top_words=10, num_top_docs=100, extra_topic=True):
        super().__init__(num_top_words, num_top_docs)
        self.vectorization_method = None
        self.extra_topic = extra_topic

    def fit(self, data, num_topics):
        self.num_topics = num_topics
        vectorizer = self.vectorization_method()
        vectors = vectorizer.fit_transform(remove_stopwords(data)).todense()
        vocab = np.array(vectorizer.get_feature_names())
        return vectors, vocab


class SVDTopicModel(MatrixTopicModel):

    def __init__(self, num_top_words=10, num_top_docs=100, extra_topic=True):
        super().__init__(num_top_words, num_top_docs, extra_topic)
        self.vectorization_method = CountVectorizer

    def fit(self, data, num_topics):
        vectors, vocab = super().fit(data, num_topics)
        U, s, Vh = linalg.svd(vectors, full_matrices=False)
        top_values = Vh[:self.num_topics]
        self.top_topics_relevance = s[:self.num_topics]
        self.top_topics = self.get_top_words(top_values, vocab)
        self.docs_topic_dist = U[:, :self.num_topics] + np.abs(np.min(U[:, :self.num_topics]))


class RandomSVDTopicModel(MatrixTopicModel):

    def __init__(self, num_top_words=10, num_top_docs=100, extra_topic=True):
        super().__init__(num_top_words, num_top_docs, extra_topic)
        self.vectorization_method = CountVectorizer

    def fit(self, data, num_topics):
        vectors, vocab = super().fit(data, num_topics)
        U, s, Vh = decomposition.randomized_svd(vectors, self.num_topics)
        self.top_topics_relevance = s
        self.top_topics = self.get_top_words(Vh, vocab)
        self.docs_topic_dist = U + np.abs(np.min(U))


class PCATopicModel(MatrixTopicModel):

    def __init__(self, num_top_words=10, num_top_docs=100, extra_topic=True):
        super().__init__(num_top_words, num_top_docs, extra_topic)
        self.vectorization_method = CountVectorizer

    def fit(self, data, num_topics):
        vectors, vocab = super().fit(data, num_topics)
        U, s, Vh = fbpca.pca(vectors, self.num_topics)
        self.top_topics_relevance = s
        self.top_topics = self.get_top_words(Vh, vocab)
        self.docs_topic_dist = U + np.abs(np.min(U))


class NMFTopicModel(MatrixTopicModel):

    def __init__(self, num_top_words=10, num_top_docs=100, extra_topic=True):
        super().__init__(num_top_words, num_top_docs, extra_topic)
        self.vectorization_method = CountVectorizer

    def fit(self, data, num_topics):
        vectors, vocab = super().fit(data, num_topics)
        clf = decomposition.NMF(n_components=self.num_topics, random_state=1)
        W1 = clf.fit_transform(vectors)
        H1 = clf.components_
        self.top_topics = self.get_top_words(H1, vocab)
        temp = np.transpose(self.top_topics, (0, 2, 1))
        self.top_topics_relevance = np.array([sum([float(s) for s in temp[i][1]]) for i in range(self.num_topics)])
        self.docs_topic_dist = W1


class NMFTFIDFTopicModel(MatrixTopicModel):

    def __init__(self, num_top_words=10, num_top_docs=100, extra_topic=True):
        super().__init__(num_top_words, num_top_docs, extra_topic)
        self.vectorization_method = TfidfVectorizer

    def fit(self, data, num_topics):
        vectors_tfidf, vocab = super().fit(data, num_topics)
        clf = decomposition.NMF(n_components=self.num_topics, random_state=1)
        W1 = clf.fit_transform(vectors_tfidf)
        H1 = clf.components_
        self.top_topics = self.get_top_words(H1, vocab)
        temp = np.transpose(self.top_topics, (0, 2, 1))
        self.top_topics_relevance = np.array([sum([float(s) for s in temp[i][1]]) for i in range(self.num_topics)])
        self.docs_topic_dist = W1


class LDATopicModel(TopicModel):

    def fit(self, data, num_topics):
        super().fit(data, num_topics)
        corpus = [row.split() for row in remove_stopwords(data)]
        dic = gensim.corpora.Dictionary(corpus)
        bow_corpus = [dic.doc2bow(doc) for doc in corpus]

        lda_model = gensim.models.LdaMulticore(bow_corpus,
                                                    num_topics=self.num_topics,
                                                    id2word=dic,
                                                    passes=10,
                                                    workers=2)
        topics = lda_model.show_topics(formatted=False)
        self.top_topics = np.array([t[1] for t in topics])
        temp = np.transpose(self.top_topics, (0, 2, 1))
        self.top_topics_relevance = np.array([sum([float(s) for s in temp[i][1]]) for i in range(self.num_topics)])
        dists = np.array(lda_model.get_document_topics(bow_corpus, minimum_probability=0.0))
        self.docs_topic_dist = dists.transpose((2, 0, 1))[1]


def get_hebrew_stopwords():
    stop_path= os.getcwd() + "/data/stopwords.txt"
    with open(stop_path, encoding="utf-8") as in_file:
        lines = in_file.readlines()
        res = [l.strip() for l in lines]
    return res


def remove_stopwords(data):
    stopwords = get_hebrew_stopwords()
    for i in range(len(data)):
        s = data[i]
        words = s.split()
        s_new = [w for w in words if w not in stopwords]
        data[i] = ' '.join(s_new)
    return data


def invert_words(words):
    return [w[::-1] for w in words]


# def show_topics(Vh_top, vocab, num_top_words=10):
#     top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
#     top_values = lambda t: np.sort(t)[:-num_top_words-1:-1]
#     topic_words = [top_words(t) for t in Vh_top]
#     values = [top_values(t) for t in Vh_top]
#     fig, axs = plt.subplots(1,4, figsize=(30,5))
#     for i in range(4):
#         axs[i].plot(invert_words(topic_words[i]), values[i], 'ro')
#     plt.show()
#     return np.array(topic_words)


def c_clusters(pred_clusters, true_clusters, K=3, score_without0=True):
    perm = np.array(list(permutations(np.arange(0, K))))
    perm_pred_clusters = np.array([[p[i] for i in pred_clusters] for p in perm])
    if score_without0:
        i = np.argmax(np.average(perm_pred_clusters[:, true_clusters != 0] == true_clusters[true_clusters != 0], axis=1))
        # print(confusion_matrix(perm_pred_clusters[i], true_clusters))
        return accuracy_score(perm_pred_clusters[i][true_clusters != 0], true_clusters[true_clusters != 0]), perm_pred_clusters[i]

    i = np.argmax([accuracy_score(p, true_clusters) for p in perm_pred_clusters])
    print(confusion_matrix(perm_pred_clusters[i], true_clusters))
    return accuracy_score(perm_pred_clusters[i], true_clusters), perm_pred_clusters[i]


def run_iterations_4_topics(data, topic_tags, num_iter=10):
    models = [SVDTopicModel, RandomSVDTopicModel, PCATopicModel, NMFTopicModel, NMFTFIDFTopicModel, LDATopicModel]
    for i in range(4):
        print(len(topic_tags[topic_tags == i]))
    for m in models:
        print(m.__name__)
        scores = []
        accuracies = []
        for j in range(num_iter):
            model = m(extra_topic=False)
            model.fit(data, 4)
            top_topics, top_topics_relevance = model.predict_topics()
            docs_topic_dist, docs_tags = model.predict_docs_dist()
            score, perm_pred = c_clusters(docs_tags, topic_tags, K=4)
            accuracy = accuracy_score(perm_pred, topic_tags)
            scores.append(score)
            accuracies.append(accuracy)
        print('non zeros score =', np.average(scores))
        print('accuracy =', np.average(accuracies))
        print('average =', (np.average(scores) + np.average(accuracies)) / 2)


def run_iterations_3_topics_extra(data, topic_tags, num_iter=10):
    models = [SVDTopicModel, RandomSVDTopicModel, PCATopicModel, NMFTopicModel, NMFTFIDFTopicModel]
    for i in range(4):
        print(len(topic_tags[topic_tags == i]))
    for m in models:
        print(m.__name__)
        scores = []
        accuracies = []
        for j in range(num_iter):
            model = m()
            model.fit(data, 3)
            top_topics, top_topics_relevance = model.predict_topics()
            docs_topic_dist, docs_tags = model.predict_docs_dist()
            score, perm_pred = c_clusters(docs_tags, topic_tags, K=4)
            accuracy = accuracy_score(perm_pred, topic_tags)
            scores.append(score)
            accuracies.append(accuracy)
        print('non zeros score =', np.average(scores))
        print('accuracy =', np.average(accuracies))
        print('average =', (np.average(scores) + np.average(accuracies)) / 2)


def check_first_step(data, topic_tags):
    models = [SVDTopicModel, RandomSVDTopicModel, PCATopicModel, NMFTopicModel, NMFTFIDFTopicModel]
    print('check best for zeros')
    for m in models:
        print(m.__name__)
        model = m()
        model.fit(data, 3)
        model.predict_topics()
        docs_topic_dist, docs_tags = model.predict_docs_dist(certainty_prec=30)

        K = 4
        i = np.argmax([sum(docs_tags == i) for i in np.arange(0, K)])
        # i -> 0, 0 -> i in docs_tags
        temp_pred = np.copy(docs_tags)
        idx0 = np.array(temp_pred == 0)
        temp_pred[temp_pred == i] = 0
        temp_pred[idx0] = i
        print("accurate zeros =", np.average(temp_pred[topic_tags == 0] == topic_tags[topic_tags == 0]))
        print("accuracy =", accuracy_score(temp_pred, topic_tags))
        print("num zeros = ", sum(temp_pred == 0))
        print("num accurate zeros / num zeros", sum(temp_pred[topic_tags == 0] == topic_tags[topic_tags == 0]) / sum(temp_pred == 0))
        # print(confusion_matrix(temp_pred, topic_tags))


def run_other_topic_first(data, algo_step1, algo_step2, certainty_prec=30):
    model = algo_step1()
    model.fit(data, 3)
    top_topics, top_topics_relevance = model.predict_topics()
    docs_topic_dist, docs_tags = model.predict_docs_dist(certainty_prec=certainty_prec)

    K = 4
    i = np.argmax([sum(docs_tags == i) for i in np.arange(0, K)])
    # i -> 0, 0 -> i in docs_tags
    temp_pred = np.copy(docs_tags)
    idx0 = np.array(temp_pred == 0)
    temp_pred[temp_pred == i] = 0
    temp_pred[idx0] = i
    # print("accurate zeros =", np.average(temp_pred[topic_tags == 0] == topic_tags[topic_tags == 0]))
    # print("accuracy =", accuracy_score(temp_pred, topic_tags))
    # print("zeros = ", sum(temp_pred == 0))
    # print(confusion_matrix(temp_pred, topic_tags))

    data_new = data[temp_pred != 0]
    model = algo_step2(extra_topic=False)
    model.fit(data_new, 3)
    top_topics, top_topics_relevance = model.predict_topics()
    docs_topic_dist, docs_tags = model.predict_docs_dist()
    pred = np.zeros(len(data), dtype=int)
    pred[temp_pred == 0] = 0
    pred[temp_pred != 0] = docs_tags + 1
    return pred


def run_iterations_2_steps(data, topic_tags, algo1, num_iter=10):
    models = [SVDTopicModel, RandomSVDTopicModel, PCATopicModel, NMFTopicModel, NMFTFIDFTopicModel]
    for i in range(4):
        print(len(topic_tags[topic_tags == i]))
    for m in models:
        print(m.__name__)
        scores = []
        accuracies = []
        for j in range(num_iter):
            pred = run_other_topic_first(data, algo_step1=algo1, algo_step2=m)
            score, perm_pred = c_clusters(pred, topic_tags, K=4)
            accuracy = accuracy_score(perm_pred, topic_tags)
            scores.append(score)
            accuracies.append(accuracy)
        print('score', np.average(scores))
        print('accuracy', np.average(accuracies))
        print('average =', (np.average(scores) + np.average(accuracies)) / 2)


def check_certainty_perc(data, topic_tags):
    models = [SVDTopicModel, RandomSVDTopicModel, PCATopicModel, NMFTopicModel, NMFTFIDFTopicModel]
    # models = [RandomSVDTopicModel, NMFTopicModel, NMFTFIDFTopicModel]
    # percs = [1, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 99]
    percs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

    TRAIN_SIZE = len(data)

    def get_folds(X_train, y_train, num_folds=10):
        fold_size = int(TRAIN_SIZE / num_folds)
        fold_idx = np.arange(fold_size, TRAIN_SIZE, fold_size)
        folds_X = []
        folds_y = []
        cur = 0
        for i in fold_idx:
            folds_X.append(X_train[cur:i])
            folds_y.append(y_train[cur:i])
            cur = i
        folds_X.append(X_train[cur:])
        folds_y.append(y_train[cur:])
        return np.array(folds_X), np.array(folds_y)

    folds_X, folds_y = get_folds(data, topic_tags)

    for m in models:
        print(m.__name__)
        mask = np.ones(len(folds_X), dtype=bool)
        scores = []
        accuracies = []
        for j in range(10):
            mask[j] = False
            cur_X_train = np.concatenate(folds_X[mask])
            cur_y_train = np.concatenate(folds_y[mask])
            mask[j] = True
            for a in percs:
                pred = run_other_topic_first(cur_X_train, algo_step1=PCATopicModel, algo_step2=m, certainty_prec=a)
                score, perm_pred = c_clusters(pred, cur_y_train, K=4)
                accuracy = accuracy_score(perm_pred, cur_y_train)
                scores.append(score)
                accuracies.append(accuracy)
        scores = np.reshape(scores, (len(percs), 10))
        accuracies = np.reshape(accuracies, (len(percs), 10))
        print('total for', m)
        print('score', np.average(scores, axis=1))
        print('accuracy', np.average(accuracies, axis=1))
        print()
        plt.plot(percs, np.average(scores, axis=1), label=str(m.__name__) + ' score')
        plt.plot(percs, np.average(accuracies, axis=1), label=str(m.__name__) + ' accuracy')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    # models = [SVDTopicModel, RandomSVDTopicModel, PCATopicModel, NMFTopicModel, NMFTFIDFTopicModel, LDATopicModel]
    # models = [SVDTopicModel, RandomSVDTopicModel, PCATopicModel, NMFTopicModel, NMFTFIDFTopicModel]
    # models = [LDATopicModel]
    # models = [PCATopicModel]
    # models = [NMFTFIDFTopicModel]
    data, topic_tags = get_sentences("final_tagged/sentences_list_shuffled", return_topic_tags=True)
    # run_iterations_4_topics(data, topic_tags, num_iter=10)
    # run_iterations_3_topics_extra(data, topic_tags, num_iter=10)
    # check_first_step(data, topic_tags)
    # run_iterations_2_steps(data, topic_tags, algo1=SVDTopicModel, num_iter=10)
    # run_iterations_2_steps(data, topic_tags, algo1=PCATopicModel, num_iter=10)
    # check_certainty_perc(data, topic_tags)
    # run_other_topic_first(data, algo_step1=PCATopicModel, algo_step2=NMFTFIDFTopicModel, certainty_prec=30)



