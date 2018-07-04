import logging as log
import sys
import urllib.parse
import urllib.request
import numpy as np

import gensim
import nltk
from nltk.corpus import stopwords as stopwords_nltk

log.basicConfig(stream=sys.stdout, level=log.DEBUG)

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


class VectorUtils:
    def __init__(self, language, redis_host='localhost', redis_port=6379,
                 redis_db=0):
        self.language = language
        self.stopwords = stopwords_nltk.words(self.language)

    def nlp_clean(self, sentence):
        preprocessed_sentence = gensim.utils.simple_preprocess(sentence)
        filtered_words = list(set(preprocessed_sentence).difference(self.stopwords))
        return ' '.join(filtered_words)

    def getVector(self, body):
        def parse_to_array(vector):
            ar = vector.decode('utf-8')
            ar = ar.replace('[', '').replace(']', '').replace('\'', '').split(',')
            return np.array([float(x) for x in ar])
        vector = self._get_sentence_vector(self.nlp_clean(body))
        return parse_to_array(vector)

    def _get_sentence_vector(self, sentence):
        encoded_sentence = urllib.parse.quote(sentence, safe='')
        return urllib.request.urlopen('http://82.192.87.234:8084/vector/array?text=%s' % encoded_sentence).read()
