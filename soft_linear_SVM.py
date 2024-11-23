# References: 
# https://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf
import csv
import pandas as pd
import numpy as np
from helper_functions.json_file import load_json
# from helper_functions.Corpus import Corpus

training_vectors_path = "./data/document_vectors_test.json"
training_data_path = "./data/training_data.csv"

class soft_linear_svm:
    
    w = np.array([[]])
    def __init__(self, C=1.0, learning_rate=0.01, epochs=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
    
        
def get_labels(corpus, criteria):
    types = corpus.type
    if criteria == "story":
        types = [1 if type == "NONELECTION" or type == "ELECTION" else -1 for type in types]
    elif criteria == "election":
        types = [1 if type == "ELECTION" else -1 for type in types]
    return types

if __name__ == "__main__":
    training_vectors = load_json(training_vectors_path)
    for vector in training_vectors:
        if not isinstance(vector, list):
            print(vector)
   
    # df = pd.read_csv(training_data_path)
    # corpus = Corpus(df)
    