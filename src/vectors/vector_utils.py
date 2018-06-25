import gensim
import nltk
import fastText as fast

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

sno = nltk.stem.SnowballStemmer("russian")
stopwords = stopwords.words("russian")


class VectorUtils:
    model = None

    def __init__(self):
        self.model = self.get_model()

    def get_model(self):
        return fast.load_model('/home/gorih/Documents/fastText/klangoo-rus.bin')

    def cleanText(self, sentence):
        tokenized_sents = word_tokenize(sentence)
        filtered_words = [sno.stem(word) for word in tokenized_sents if word not in stopwords]
        return ' '.join(filtered_words)

    def getVector(self, body):
        words = gensim.utils.simple_preprocess(body)
        vector = self.model.get_sentence_vector(self.cleanText(str(' '.join(words))))
        print("Vector of : " + body)
        print(vector)
        return vector