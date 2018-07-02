import io
import redis
import numpy as np
import gensim
import nltk
import logging as log
from nltk.corpus import stopwords as stopwords_nltk
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


class VectorUtils:
    data = None
    language = None
    redis = None

    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        if not VectorUtils.redis: VectorUtils.redis = redis.Redis(redis_host, redis_port, redis_db)
        self.sno = nltk.stem.SnowballStemmer(VectorUtils.language)
        self.stopwords = stopwords_nltk.words(VectorUtils.language)

    @classmethod
    def load_vocabulary(cls, vectors_model_path, redis_host='localhost', redis_port=6379, redis_db=0):
        log.info("Started loading of vocabulary.")
        VectorUtils.redis = redis.Redis(redis_host, redis_port, redis_db)
        fin = io.open(vectors_model_path, 'r', encoding='utf-8',
                      newline='\n', errors='ignore')
        for line in fin:
            tokens = line.rstrip().split(' ')
            log.debug("Saving word ' %s ' to vocabulary" % tokens[0])
            VectorUtils.redis.set(tokens[0], tokens[1:])

    def cleanText(self, sentence):
        tokenized_sents = word_tokenize(sentence)
        filtered_words = [self.sno.stem(word) for word in tokenized_sents if word not in self.stopwords]
        return ' '.join(filtered_words)

    def getVector(self, body):
        cleaned_text = self.cleanText(body).lower()
        words = gensim.utils.simple_preprocess(cleaned_text)
        vectors = np.zeros(300)
        for word in words:
            current_vector = self._words_vector(word)
            vectors = self.sum_vectors(vectors, current_vector)

        return vectors

    def _words_vector(self, word):
        w = VectorUtils.redis.get(word)
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
