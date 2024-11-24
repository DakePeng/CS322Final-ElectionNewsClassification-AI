import pandas as pd
import numpy as np
from helper_functions.json_file import load_json
from helper_functions.Corpus import Corpus
from sklearn.svm import LinearSVC
from tqdm import tqdm

training_vectors_path = "./data/document_vectors_training.json"
training_data_path = "./data/training_data.csv"

test_data_path = "./data/test_data.csv"
test_vectors_path = "./data/document_vectors_test.json"
import numpy as np

class Soft_Margin_Linear_SVM:
    
    '''references: tutorials from
    https://www.geeksforgeeks.org/implementing-svm-from-scratch-in-python/
    https://www.youtube.com/watch?v=WdXapAG6TYo
    '''
    def __init__(self, lambda_param = 0.01, learning_rate=0.001, num_iters = 1000):
        # the smaller the lambda, the less tolerant the model is to misclassification
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.w = None
        self.b = None
        
    def fit(self, x, y):
        num_samples, num_features = x.shape
        # the plane is defined by wx + b = 0
        self.w = np.zeros(num_features)
        self.b = 0
        print("Training Model...")
        for i in tqdm(range(self.num_iters)):
            self.step(x, y)
        
    def step(self, x, y):
        for i in range(len(x)):
            xi = x[i]
            yi = y[i]
            
            # gradient descent:
            margin = yi * (np.dot(xi, self.w) - self.b)
        
            # if fit: use the hard margin rewards
            delta_w = 2 * self.lambda_param * self.w
            delta_b = 0
            
            # if not fit:
            if margin < 1:
                # for vectors that are misfit (dot product is negative), include penalty
                delta_w -= np.dot(xi, yi)
                delta_b = yi 
            
            # update values
            self.w -= self.learning_rate * delta_w 
            self.b -= self.learning_rate * delta_b

    def predict(self, x):
        results = np.dot(x, self.w) - self.b
        return np.sign(results)

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
    training_vectors = np.array([training_vectors[i] for i in story_indices])
    training_labels = np.array(get_labels(corpus, "election"))
    return training_vectors, training_labels

def predict(df_test, document_vectors_test, model_story, model_election):
    results_story = model_story.predict(document_vectors_test)
    story_indices = []
    for i in range(len(results_story)):
        if results_story[i] == 1:
            story_indices.append(i)
    story_vectors = document_vectors_test[results_story == 1]
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
    
    x1, y1 = get_training_set_story(training_vectors, corpus)
    # model_story = LinearSVC(dual=False, C=1, max_iter=1000) # sklearn implementation
    # the story-nonstory distinction is more rigid
    model_story = Soft_Margin_Linear_SVM() 
    model_story.fit(x1, y1)
    
    x2, y2 = get_training_set_election(training_vectors, corpus)
    # model_election = LinearSVC(dual=False, C=1, max_iter=1000)
    model_election = Soft_Margin_Linear_SVM()
    model_election.fit(x2, y2)

    df_test = pd.read_csv(test_data_path)
    document_vectors_test = get_document_vectors_from_json(test_vectors_path)
    predict(df_test, document_vectors_test, model_story, model_election)