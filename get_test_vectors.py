import pandas as pd
import gensim.downloader as api
from helper_functions.json_file import load_json, save_json
from helper_functions.Corpus import Corpus
from helper_functions.gensim_model import word2vec_model
from get_tf_idf import get_tf_str, get_tf_idf_weight
from get_training_vectors import get_document_vectors_json

idf_data_path = "./data/idf_training.json"
test_data_path = "./data/test_data.csv"
test_document_vectors_path  = "./data/document_vectors_test.json"
test_weights_path = "./data/tf_idf_weights_test.json"

dev_data_path = "./data/dev_data.csv"
dev_document_vectors_path  = "./data/document_vectors_dev.json"
dev_weights_path = "./data/tf_idf_weights_dev.json"


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

def get_test_dev_vectors(data_path, weights_output_path, document_vectors_output_path):
    df = pd.read_csv(data_path)
    corpus = Corpus(df)
    idf = load_json(idf_data_path)
    generate_test_weights(corpus, idf, weights_output_path)
    get_document_vectors_json(word2vec_model, data_path, weights_output_path, document_vectors_output_path)  
           
if __name__ == "__main__":
    get_test_dev_vectors(test_data_path, test_weights_path, test_document_vectors_path)
    get_test_dev_vectors(dev_data_path, dev_weights_path, dev_document_vectors_path)