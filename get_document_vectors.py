import time
import gensim.downloader as api
import pandas as pd
from helper_functions.Corpus import Corpus
from get_tf_idf import load_weights

training_data_path = "./data/training_data.csv"
weights_data_path = "./data/tf_idf_weights.json"
def get_document_vectors(model, words, weights, document_index):
    weighted_vectors = []
    for word in words:
        word_vec = model.wv[word]
        weighted_vectors.append(word_vec * weights(document_index))
    return


if __name__ == "__main__":
    print('Loading Data...')
    start = time.time()
    model = api.load('word2vec-google-news-300')
    weights = load_weights(weights_data_path)
    df = pd.read_csv(training_data_path)
    corpus = Corpus(df)
    print('Done. ({} seconds)'.format(time.time() - start))