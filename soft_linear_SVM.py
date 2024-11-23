# References: 
# https://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf
import csv
import pandas as pd
import numpy as np
from helper_functions.json_file import load_json
from helper_functions.Corpus import Corpus
from sklearn.svm import LinearSVC
import time

training_vectors_path = "./data/document_vectors_training.json"
training_data_path = "./data/training_data.csv"

test_data_path = "./data/test_data.csv"
test_vectors_path = "./data/document_vectors_test.json"

class soft_linear_svm:
    
    def __init__(self, C=1.0, learning_rate=0.01, epochs=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def train(self):
        return
    
def get_labels(corpus, criteria):
    types = corpus.type
    if criteria == "story":
        types = [1 if (type == "NONELECTION" or type == "ELECTION") else -1 for type in types]
    elif criteria == "election":
        types = [1 if type == "ELECTION" else -1 for type in types]
    return types

def get_document_vectors_from_json(path):
    return np.array(load_json(path))

def get_training_set_story(training_vectors, corpus):
    training_labels = np.array(get_labels(corpus, "story"))
    return training_vectors, training_labels

def get_training_set_election(training_vectors, corpus):
    story_indices = corpus.remove_none()
    training_vectors = [training_vectors[i] for i in story_indices]
    training_labels = np.array(get_labels(corpus, "election"))
    return training_vectors, training_labels

def predict(df_test, document_vectors_test, model_story, model_election):
    results_story = model_story.predict(document_vectors_test)
    story_indices = []
    for i in range(len(results_story)):
        if results_story[i] == 1:
            story_indices.append(i)
    story_vectors = [document_vectors_test[i] for i in story_indices]
    results_election = model_election.predict(story_vectors)
    election_indices = []
    for i in range(len(results_election)):
        if results_election[i] == 1:
            election_indices.append(i)
            
    # write results to csv
    df_test['SVM_Prediction'] = None
    for index, row in df_test.iterrows():
        if index in election_indices:
            df_test.loc[index, 'SVM_Prediction'] = "ELECTION"
        elif index in story_indices:
            df_test.loc[index, 'SVM_Prediction'] = "NONELECTION"
        else: 
            df_test.loc[index, 'SVM_Prediction'] = "NONE"
    df_test.to_csv('./results/SVM_Predictions.csv', index=False)

if __name__ == "__main__":
    
    training_vectors = get_document_vectors_from_json(training_vectors_path)
    df = pd.read_csv(training_data_path)
    corpus = Corpus(df)
    
    t = time.time()
    x1, y1 = get_training_set_story(training_vectors, corpus)
    model_story = LinearSVC(dual=False, C=1, max_iter=1000)
    model_story.fit(x1, y1)
    
    x2, y2 = get_training_set_election(training_vectors, corpus)
    model_election = LinearSVC(dual=False, C=1, max_iter=1000)
    model_election.fit(x2, y2)

    df_test = pd.read_csv(test_data_path)
    document_vectors_test = get_document_vectors_from_json(test_vectors_path)
    predict(df_test, document_vectors_test, model_story, model_election)