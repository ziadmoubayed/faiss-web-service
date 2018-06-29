import io
import redis
import numpy as np
import gensim
import nltk
from nltk.corpus import stopwords as stopwords_nltk
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

class VectorUtils:
    data = None
    sno = None
    stopwords = None

    def __init__(self):
        self.redis = redis.Redis()

    @staticmethod
    def init(vectors_model_path, language):
        VectorUtils.sno = nltk.stem.SnowballStemmer(language)
        VectorUtils.stopwords = stopwords_nltk.words(language)

    def cleanText(self, sentence):
        tokenized_sents = word_tokenize(sentence)
        filtered_words = [VectorUtils.sno.stem(word) for word in tokenized_sents if word not in VectorUtils.stopwords]
        return ' '.join(filtered_words)

    def getVector(self, body):
        cleaned_text = self.cleanText(body).lower()
        words = gensim.utils.simple_preprocess(cleaned_text)
        vectors = np.zeros(300)
        for word in words:
            current_vector = self.words_vecotr(word)
            vectors = self.sum_vectors(vectors, current_vector)

        return vectors

    def words_vecotr(self, word):
        w = self.redis.get(word)
        if w:
            ar = w.decode('utf-8')
            ar = ar.replace('[', '').replace(']', '').replace('\'', '').split(', ')
            return np.array([float(x) for x in ar])
        return np.zeros(300)

    def sum_vectors(self, a, b):
        c = np.zeros(a.__len__())
        for i in range(0, a.__len__()):
            c[i] = a[i] + b[i]
        return c