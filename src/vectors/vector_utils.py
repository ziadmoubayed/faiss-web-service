import logging as log
import string
import sys

import nltk
import numpy as np
import redis
from nltk.corpus import stopwords as stopwords_nltk
from nltk.tokenize import word_tokenize

log.basicConfig(stream=sys.stdout, level=log.DEBUG)

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

class VectorUtils:
    data = None
    language = "english"
    redis = None

    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        if not VectorUtils.redis:
            VectorUtils.redis = redis.Redis(redis_host, redis_port, redis_db)
        self.sno = nltk.stem.SnowballStemmer(VectorUtils.language)
        self.stopwords = stopwords_nltk.words(VectorUtils.language)
        self.translator = str.maketrans('', '', string.punctuation)

    @staticmethod
    def load_vocabulary(vectors_model_path, redis_host='localhost', redis_port=6379, redis_db=0):
        log.info("Started loading of vocabulary.")
        VectorUtils.redis = redis.Redis(redis_host, redis_port, redis_db)
        with open(vectors_model_path) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                tokens = line.rstrip().split(' ')
                word = tokens[0]
                log.debug("Saving word ' %s ' to vocabulary. Loaded %d" % (word, cnt))
                VectorUtils.redis.set(word, tokens[1:])
                cnt += 1

        log.info("Finished vocabulary loading. Loaded %d" % cnt)

    def nlp_clean(self, sentence):
        string_name = sentence.translate(self.translator)
        new_str = string_name.lower()
        # print(sentence)
        words = word_tokenize(new_str)
        # print(words)
        words = list(set(words).difference(self.stopwords))
        return words

    def getVector(self, body):
        words = self.nlp_clean(body)
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
