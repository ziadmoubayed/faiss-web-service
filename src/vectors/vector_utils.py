import fastText as fast
import gensim
import nltk
from nltk.corpus import stopwords as stopwords_nltk
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

class VectorUtils:
    sno = None
    stopwords = None
    model = None

    @staticmethod
    def init(vectors_model_path, language):
        if not VectorUtils.model:
            VectorUtils.model = fast.load_model(vectors_model_path)
            VectorUtils.sno = nltk.stem.SnowballStemmer(language)
            VectorUtils.stopwords = stopwords_nltk.words(language)

    def cleanText(self, sentence):
        tokenized_sents = word_tokenize(sentence)
        filtered_words = [VectorUtils.sno.stem(word) for word in tokenized_sents if word not in VectorUtils.stopwords]
        return ' '.join(filtered_words)

    def getVector(self, body):
        words = gensim.utils.simple_preprocess(body)
        vector = VectorUtils.model.get_sentence_vector(self.cleanText(str(' '.join(words))))
        print("Vector of : " + body)
        print(vector)
        return vector