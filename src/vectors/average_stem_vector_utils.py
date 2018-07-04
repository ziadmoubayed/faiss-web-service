import logging as log
import string
import sys

import nltk
import numpy as np
import redis
import urllib.request

from nltk.corpus import stopwords as stopwords_nltk
from nltk.tokenize import word_tokenize
from vocabulary.vocabulary import VocabularyKeeper

log.basicConfig(stream=sys.stdout, level=log.DEBUG)

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


class VectorUtils:
    def __init__(self, language, redis_host='localhost', redis_port=6379,
                 redis_db=0):
        self.language = language
        self.redis = redis.Redis(redis_host, redis_port, redis_db)
        self.sno = nltk.stem.SnowballStemmer(self.language)
        self.stopwords = stopwords_nltk.words(self.language)
        self.translator = str.maketrans('', '', string.punctuation)

    def nlp_clean(self, sentence):
        string_name = sentence.translate(self.translator)
        new_str = string_name.lower()
        words = word_tokenize(new_str)
        stemmed_words = [self.sno.stem(word) for word in words]
        words = list(set(stemmed_words).difference(self.stopwords))
        return words

    # gets vector of each word in text
    # and returns sum of all vectors
    def getVector(self, body):
        words = self.nlp_clean(body)
        doc = [self._words_vector(word) for word in words]
        return np.mean(doc, axis=0)

    # checks for word in vocabulary
    # if word was found - returns it's vector
    # otherwise makes http call to API to get vector, saves result to vocabulary
    # and returns vector
    # if call to API was failed, vector filled with zeros will be returned
    def _words_vector(self, word):
        def parse_to_array(vector):
            ar = vector.decode('utf-8')
            ar = ar.replace('[', '').replace(']', '').replace('\'', '').split(',')
            return np.array([float(x) for x in ar])

        if VocabularyKeeper.in_memory:
            w = VocabularyKeeper.vocabulary[word]
            if w:
                return np.array(list(w))
        else:
            w = self.redis.hget('vocabulary', word)
            if w:
                return parse_to_array(w)

        log.debug("Word ' %s ' was not found in vocabulary." % word)
        try:
            raw_vector = self._get_unknown_word_vector(word)

            new_words_vector = parse_to_array(raw_vector)
            if VocabularyKeeper.in_memory:
                VocabularyKeeper.vocabulary[word] = new_words_vector
            else:
                self.redis.hset('vocabulary', word, raw_vector)

            return new_words_vector
        except:
            log.info("Failed to call API for getting vector for word :' %s '." % word)

        return np.zeros(300)


    def _get_unknown_word_vector(self, word):
        return urllib.request.urlopen('http://82.192.87.234:8084/vector/array?text=%s' % word).read()
