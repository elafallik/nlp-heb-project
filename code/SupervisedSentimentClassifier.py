from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from data_preparation import get_sentences, save_sentiment_pred, sort_sentences_list, save_sentiment_pred_by_post
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def eval_accuracy(test_tags, prediction):
    # print(accuracy_score(test_tags, prediction))
    print(confusion_matrix(test_tags, prediction))
    return accuracy_score(test_tags, prediction)


class SupervisedSentimentClassifier:

    def __init__(self):
        self.classifier_model = None
        self.count_vec = None
        self.tf_transformer = None
        self.clf = None

    def fit(self, train_data, train_tags, use_class_weight=False):
        """
        :param train_data: list of sentences splits to morphemes
        :param test_data:  list of sentences splits to morphemes
        :param train_tags: tags for each sentence in train_data.
                Possible tags = 1 positive
                                0 neutral
                                -1 negative
        """
        self.count_vec = CountVectorizer()
        X_train_counts = self.count_vec.fit_transform(train_data)
        # print("X_train_counts.shape:", X_train_counts.shape)
        self.tf_transformer = TfidfTransformer().fit(X_train_counts)
        X_train_tfidf = self.tf_transformer.transform(X_train_counts)
        inverse_dict = {self.count_vec.vocabulary_[w]: w for w in self.count_vec.vocabulary_.keys()}
        if use_class_weight:
            self.clf = self.classifier_model(class_weight='balanced').fit(X_train_tfidf, train_tags)
        else:
            self.clf = self.classifier_model().fit(X_train_tfidf, train_tags)

    def predict(self, test_data):
        X_test_counts = self.count_vec.transform(test_data)
        # print("X_test_counts.shape:", X_test_counts.shape)
        X_test_tfidf = self.tf_transformer.transform(X_test_counts)
        return self.clf.predict(X_test_tfidf)


class BayesSentimentClassifier(SupervisedSentimentClassifier):

    def __init__(self):
        SupervisedSentimentClassifier.__init__(self)
        self.classifier_model = MultinomialNB


class LogisticRegressionSentimentClassifier(SupervisedSentimentClassifier):

    def __init__(self):
        SupervisedSentimentClassifier.__init__(self)
        self.classifier_model = LogisticRegression


class RandomForestSentimentClassifier(SupervisedSentimentClassifier):

    def __init__(self):
        SupervisedSentimentClassifier.__init__(self)
        self.classifier_model = RandomForestClassifier



def get_folds(X_train, y_train, n_folds):
    train_size = len(X_train)
    fold_size = int(train_size / n_folds)
    fold_idx = np.arange(fold_size, train_size, fold_size)
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


def run_algo(model, train_tokens, train_tags, test_tokens, test_tags):
    model.fit(train_tokens, train_tags)
    predicted = model.predict(test_tokens)
    # print(confusion_matrix(test_tags, predicted))
    return accuracy_score(test_tags, predicted)


def CV(m, data, tags, n_folds, use_class_weight=False):
    folds_X, folds_y = get_folds(data, tags, n_folds)
    mask = np.ones(len(folds_X), dtype=bool)
    accuracies = []
    for j in range(n_folds):
        mask[j] = False
        cur_X_train = np.concatenate(folds_X[mask])
        cur_y_train = np.concatenate(folds_y[mask])
        cur_X_test = folds_X[j]
        cur_y_test = folds_y[j]
        # print(data.shape, cur_X_train.shape, cur_X_test.shape)
        mask[j] = True
        if use_class_weight:
            model = m(class_weight='balanced')
        else:
            model = m()
        accuracy = run_algo(model, cur_X_train, cur_y_train, cur_X_test, cur_y_test)
        accuracies.append(accuracy)
    print('accuracy', np.average(accuracies))


def run_CV(data, data_tags, n_folds=10):
    for i in [0, 1, 2]:
        print(len(data_tags[data_tags == i]))

    count_vec = CountVectorizer()
    X_train_counts = count_vec.fit_transform(data)
    # print("X_train_counts.shape:", X_train_counts.shape)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tf_transformer.transform(X_train_counts)
    inverse_dict = {count_vec.vocabulary_[w]: w for w in count_vec.vocabulary_.keys()}

    models = [MultinomialNB, LogisticRegression, RandomForestClassifier]
    for m in models:
        print(m.__name__)
        CV(m, X_train_tfidf.toarray(), data_tags, n_folds)
    models = [LogisticRegression, RandomForestClassifier]
    for m in models:
        print(m.__name__ + ' balanced')
        CV(m, X_train_tfidf.toarray(), data_tags, n_folds, use_class_weight=True)


if __name__ == '__main__':
    # train_data, train_sentiment_tags = get_sentences('final_tagged/sentences_list_shuffled_tagged_topics_2', return_sentiment_tags=True)
    # train_sentiment_tags += 1
    # run_CV(train_data, train_sentiment_tags, n_folds=10)
    sentences_list = 'final_tagged/sentences_list_shuffled_tagged_topics_3'
    train_data, train_sentiment_tags = get_sentences(sentences_list + '_train', return_sentiment_tags=True)
    train_sentiment_tags += 1

    test_data, test_sentiment_tags = get_sentences(sentences_list + '_test', return_sentiment_tags=True)
    test_sentiment_tags += 1

    model = LogisticRegressionSentimentClassifier()
    model.fit(train_data, train_sentiment_tags, use_class_weight=True)
    pred = model.predict(test_data)
    save_sentiment_pred(sentences_list + '_test', pred - 1)
    sort_sentences_list(sentences_list + '_test_pred_sentiment')
    save_sentiment_pred_by_post(sentences_list + '_test_pred_sentiment_sorted', 'final_tagged/posts_list', '3')
