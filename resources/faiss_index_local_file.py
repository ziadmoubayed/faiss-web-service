def GET_FAISS_INDEX():
    import faiss
    return faiss.read_index(INDEX_FILE_PATH)

def GET_FAISS_ID_TO_VECTOR():
    import pickle

    with open(IDS_MAP_FILE_PATH, 'rb') as f:
        ids_vectors = pickle.load(f)

    def id_to_vector(id_):
        try:
            return ids_vectors[id_]
        except:
            pass

    return id_to_vector

INDEX_FILE_PATH = "/home/gorih/PycharmProjects/faiss-web-service/resources/index"

IDS_MAP_FILE_PATH = "/home/gorih/PycharmProjects/faiss-web-service/resources/ids_vectors.p"

# UPDATE_FAISS_AFTER_SECONDS = 60

FASTTEXT_MODEL_PATH = "/home/gorih/Documents/fastText/klangoo-rus.bin"

LANGUAGE = "russian"

INDEX_DIMENSIONS = 50

INDEX_INPUT_QUEUE = "doc2vector"