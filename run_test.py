import pandas as pd
import gensim.downloader as api
from helper_functions.json_file import load_json, save_json
from helper_functions.Corpus import Corpus
from helper_functions.gensim_model import word2vec_model
from get_tf_idf import get_tf_str, get_tf_idf_weight
from get_document_vectors import get_document_vectors_json

idf_data_path = "./data/idf_training.json"
test_data_path = "./data/test_data.csv"
test_document_vectors_path  = "./data/document_vectors_test.json"
test_weights_path = "./data/tf_idf_weights_test.json"

def generate_test_document_vectors():
    return 

def generate_test_weights(corpus, idf, output):
    documents = corpus.corpus
    tf = []
    for words in documents:
        tf_doc = get_tf_str(words)
        tf.append(tf_doc)
    weights = get_tf_idf_weight(tf, idf)
    save_json(weights, output)
                
if __name__ == "__main__":
    df = pd.read_csv(test_data_path)
    test_corpus = Corpus(df)
    idf = load_json(idf_data_path)
    generate_test_weights(test_corpus, idf, test_weights_path)
    get_document_vectors_json(word2vec_model, test_data_path, test_weights_path, test_document_vectors_path)