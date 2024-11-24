import time
import gensim.downloader as api
import numpy as np
print("Loading gensim model...")
word2vec_model = api.load('word2vec-google-news-300')
print("Done")
zerovector = [0] * 300
