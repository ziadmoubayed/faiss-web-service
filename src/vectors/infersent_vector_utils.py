
import logging as log
import sys
import urllib.parse
import urllib.request
import numpy as np

import gensim
import nltk
from nltk.corpus import stopwords as stopwords_nltk
from vocabulary.infer_sent import InferSentModelKeeper

log.basicConfig(stream=sys.stdout, level=log.DEBUG)

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


class VectorUtils:
    def __init__(self, language, redis_host='localhost', redis_port=6379,
                 redis_db=0):
        self.language = language
        self.sno = nltk.stem.SnowballStemmer(self.language)
        self.stopwords = stopwords_nltk.words(self.language)

    def nlp_clean(self, sentence):
        preprocessed_sentence = gensim.utils.simple_preprocess(sentence)
        filtered_words = [self.sno.stem(word) for word in preprocessed_sentence if word not in self.stopwords]
        return ' '.join(filtered_words)

    def getVector(self, body):
        cleaned = self.nlp_clean(body)
        InferSentModelKeeper.model.update_vocab([body])
        vector = InferSentModelKeeper.model.encode([cleaned])[0]

        return vector