from collections import defaultdict

import jieba
import jieba.posseg
import numpy as np
import pandas as pd

DEFAULT_IDF = 'default_median'

from gensim.models.keyedvectors import KeyedVectors

# file = r"45000-small.txt"
# wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False)
# print(222222222222222222)


def generate_idf_dict(documents):
    """逆文档对数频率"""
    counter = defaultdict(int)
    for documemt in documents:
        for word in set(jieba.cut(documemt)):
            counter[word] += 1

    words, counts = zip(*(counter.items()))
    n_documents = len(documents)

    words = list(words) + [DEFAULT_IDF]
    counts = np.array(list(counts) + [np.median(counts)], dtype=np.float64)

    # smoothing
    log_idf = np.log(n_documents + 1) - np.log(counts + 1) + 1

    return dict(zip(words, log_idf))


def generate_tf_dict(documents, l2_norm=True):
    """词频"""
    counter = defaultdict(int)
    for documemt in documents:
        for word in set(jieba.cut(documemt)):
            counter[word] += 1

    words, counts = zip(*(counter.items()))
    n_documents = len(documents)

    probs = np.array(counts, dtype=np.float64) / n_documents
    if l2_norm:
        probs /= np.linalg.norm(probs)

    return dict(zip(words, probs))


def tf_idf(word, tf_dict, idf_dict):
    """TF-IDF值"""
    tf_value = tf_dict.get(word, 0.0)
    idf_value = idf_dict.get(word, idf_dict[DEFAULT_IDF])

    return tf_value * idf_value, tf_value, idf_value


with open(r"C:\Users\merlin\Desktop\train_sentiment.txt", encoding='utf8') as f:
    sentences = [line.strip().split('\t')[1] for line in f.readlines()]
    idf_dict = generate_idf_dict(sentences)
    for i in sorted(idf_dict.items(), key=lambda x:x[1])[:1000]:
        if len(i[0]) > 1:
            print(i)
    raise ValueError

with open(r"C:\Users\merlin\Desktop\test_sentiment.txt", encoding='utf8') as f:
    sentences = [line.strip().split('\t')[1] for line in f.readlines()]
    tf_dict = generate_tf_dict(sentences)

    words = []
    for sentence in sentences:
        for word, tag in jieba.posseg.cut(sentence):
            tf_idf_value, tf_value, idf_value = tf_idf(word, tf_dict, idf_dict)
            words.append((word, tag, tf_idf_value, tf_value, idf_value))
    words = set(words)

    # features = []
    # features_names = ['word', 'tag', 'tf_idf', 'tf', 'idf'] + [
    #     'vec_%d' % i for i in range(200)]
    # for w in sorted(words, key=lambda x: x[3]):
    #     if len(w[0]) > 1 and w[1].startswith('n'):
    #         if w[0] in wv_from_text:
    #             vec = wv_from_text[w[0]]
    #         else:
    #             print(w)
    #             vec = np.random.randn(200)
    #         features.append(dict(zip(features_names, list(w) + list(vec))))
    # pd.DataFrame(features).to_csv('hot_words_dataset.csv', encoding='gbk', index=False)
