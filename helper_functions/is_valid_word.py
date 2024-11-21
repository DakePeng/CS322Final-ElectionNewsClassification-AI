# https://chatgpt.com/share/673eaf39-1414-800f-966e-7033f1bf1678
import nltk
import gensim.downloader as api
from nltk.corpus import words, stopwords

# uncomment if you don't have the corpus downloaded
model = api.load('word2vec-google-news-300')
nltk.download('stopwords')

# Load the English word list and stop word list
stop_words = set(stopwords.words('english'))

def is_valid_word(word):
    # Ensure the word is in lowercase for matching
    if not word:
        return False
    word_lower = word.lower()
    return word_lower in model and word_lower not in stop_words
