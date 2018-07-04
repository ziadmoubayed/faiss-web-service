INDEX_FILE_PATH = "/home/gorih/PycharmProjects/faiss-web-service/resources/index"
IDS_MAP_FILE_PATH = "/home/gorih/PycharmProjects/faiss-web-service/resources/ids_vectors.p"

### If you going to use average vectors logic (average_stem_vector_utils or average_vector_utils)
# configure three below properties
# set LOAD_VOCABULARY flag to True if vocabulary (word's vectors mapping) should be loaded on application start
# otherwise all word's vectors will be taken by calling API
LOAD_VOCABULARY = False
# set VOCABULARY_IN_MEMORY to store vocabulary in memory. !!! Be careful vocabulary could be pretty heavy
VOCABULARY_IN_MEMORY = False
# path to file, where <voc_file_name>.vec is stored. IF LOAD_VOCABULARY is set to False - this path could be skipped
WORDS_VECTORS_FILE_PATH = "/root/fws/fast_text_models/wiki_vec"

# set LOAD_DOC2VEC_MODEL to True if you want to use doc2vec model for getting vector of documents
# if this model will not be loaded application makes API calls, to get vector for each document.
LOAD_DOC2VEC_MODEL = False
DOC2VEC_MODEL_FILE_PATH = ""

# language of incoming text's
LANGUAGE = "english"
INDEX_DIMENSIONS = 300

# queue from which index will be filled
INDEX_INPUT_QUEUE = "doc2vector"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 2

INDEX_WRITES_FREQUENCY_SEC = 60
APP_HOST = "localhost"
APP_PORT = 5002

# If true - application will persist in redis each uuid - body pair
# this is needed to be able to return similarity results with bodies (not only ids)
PERSIST_BODIES = True

