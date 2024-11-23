
import numpy as np
import pandas as pd 
from tqdm import tqdm 
from helper_functions.Corpus import Corpus
from helper_functions.json_file import load_json, save_json
from helper_functions.gensim_model import word2vec_model, zerovector

training_data_path = "./data/training_data.csv"
weights_data_path = "./data/tf_idf_weights_training.json"
document_vectors_path = "./data/document_vectors_training.json"

def get_document_vector(model, words, weights, document_index):
    weighted_vectors = []
    for word in words:
        word_vec = np.array(model[word])
        weights_doc = weights[document_index]
        if word in weights_doc:
            # consider new words in test cases
            weighted_vectors.append(word_vec * weights_doc[word])
    if len(weighted_vectors) == 0:
        return zerovector
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

def get_document_vectors_json(model, data_path, weights_path, output_path):
    df = pd.read_csv(data_path)
    weights = load_json(weights_path)
    corpus = Corpus(df)
    save_json(get_document_vectors(model, corpus, weights), output_path)
    
if __name__ == "__main__":
    get_document_vectors_json(word2vec_model, training_data_path, weights_data_path, document_vectors_path)