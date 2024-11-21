import re
from helper_functions.is_valid_word import is_valid_word

class Corpus():
    corpus = []
    type = []
    def __init__(self, df):
        corpus, type = get_corpus_and_type(df)
        self.type = type
        self.corpus = clean_corpus(corpus)
        
def get_corpus_and_type(df):
    corpus = [None] * len(df)
    type = [None] * len(df)
    for index, row in df.iterrows():
        text = row["Text"].lower()
        type_row = row["Type"]
        corpus[index] = text
        type[index] = type_row
    return corpus, type

def clean_corpus(corpus):
    cleaned_corpus = [None] * len(corpus)
    for i in range(len(corpus)):
        text = corpus[i]
        cleaned_corpus[i] = get_clean_valid_words(text)
    return cleaned_corpus

def clean_word(word):
    # Remove word initial and final punctuations
    word_cleaned = None
    results = re.findall(r"[a-zA-Z0-9']+", word)
    if results:
        word_cleaned = results[0]
    return word_cleaned

def get_clean_valid_words(text):
    cleaned_words = []
    words = text.split()
    for word in words:
        cleaned_word = clean_word(word)
        if is_valid_word(cleaned_word):
            cleaned_words.append(cleaned_word)
    return cleaned_words
