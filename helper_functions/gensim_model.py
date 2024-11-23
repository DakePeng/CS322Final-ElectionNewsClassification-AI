import time
import gensim.downloader as api
import numpy as np
word2vec_model = api.load('word2vec-google-news-300')
zerovector = [0] * 300