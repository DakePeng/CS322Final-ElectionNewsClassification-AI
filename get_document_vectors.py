import time
import gensim.downloader as api
import numpy as np
import pandas as pd 
from tqdm import tqdm 
from helper_functions.Corpus import Corpus
from helper_functions.json_file import load_json, save_json

training_data_path = "./data/training_data.csv"
weights_data_path = "./data/tf_idf_weights_training.json"
document_vectors_path = "./data/document_vectors_training.json"

def get_document_vector(model, words, weights, document_index):
    weighted_vectors = []
    for word in words:
        word_vec = np.array(model[word])
        weighted_vectors.append(word_vec * weights[document_index][word])
    weighted_vector = np.mean(weighted_vectors, axis = 0)
    return weighted_vector.tolist()

def get_document_vectors(model, corpus, weights):
    document_vectors = []
    documents = corpus.corpus
    for i in tqdm(range(len(documents))):
        words = documents[i]
        document_vector = get_document_vector(model, words, weights, i)
        document_vectors.append(document_vector)
    return document_vectors

if __name__ == "__main__":
    print('Loading Data...')
    start = time.time()
    model = api.load('word2vec-google-news-300')
    weights = load_json(weights_data_path)
    df = pd.read_csv(training_data_path)
    corpus = Corpus(df)
    print('Done. ({} seconds)'.format(time.time() - start))
    save_json(get_document_vectors(model, corpus, weights), document_vectors_path)