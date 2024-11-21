import sys
import pandas as pd
import math

from collections import defaultdict as dd
from helper_functions.Corpus import Corpus
from helper_functions.json_file import save_json

training_data_path = "./data/training_data.csv"
weight_path = "./data/tf_idf_weights_training.json"
idf_path = "./data/idf_training.json"

def get_tf(corpus):
    tf = []
    document_list = corpus.corpus
    for i in range(len(document_list)):
        document_words = document_list[i]
        tf_doc = get_tf_str(document_words)
        tf.append(tf_doc)
    return squash_tf_log(tf)

def squash_tf_log(tf):
    for i in range(len(tf)):
        tf_doc = tf[i]
        for key in tf_doc:
            if tf_doc == 0: 
                continue
            else:
                tf_doc[key] = 1 + math.log10(tf_doc[key])
        tf[i] = tf_doc
    return tf

def get_tf_str(words):
    tf_single = dd(float)
    for word in words:
        tf_single[word] += 1
    return tf_single

def get_docfreq(corpus):
    word_dict = dd(float)
    document_list = corpus.corpus
    for words in document_list:
        unique_words = set(words)
        for word in unique_words:
            word_dict[word] += 1
    return word_dict

def get_idf(corpus):
    n = len(corpus.corpus)
    df = get_docfreq(corpus)
    idf = dd(float)
    for key in df:
        idf[key] = math.log10(n/df[key])
    return idf

def get_tf_idf_weight(tf, idf):
    weight = []
    for i in range(len(tf)):
        weight_doc = dd(float)
        tf_doc = tf[i]
        for key in tf_doc:
            if key in idf:
                # in the test cases there may be new words
                weight_doc[key] = tf_doc[key] * idf[key]
        weight.append(weight_doc)
    return weight

if __name__ == "__main__":
    df = pd.read_csv(training_data_path)
    corpus = Corpus(df)
    tf = get_tf(corpus)
    idf = get_idf(corpus)
    weights = get_tf_idf_weight(tf, idf)
    save_json(weights, weight_path)
    save_json(idf, idf_path)