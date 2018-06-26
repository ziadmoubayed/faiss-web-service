DEBUG = False

def GET_FAISS_RESOURCES():
    return None

def GET_FAISS_INDEX():
    import faiss

    index_file_path = '/tmp/faiss-web-service/index'
    return faiss.read_index(index_file_path)

def GET_FAISS_ID_TO_VECTOR():
    import pickle

    ids_vectors_path = '/tmp/faiss-web-service/ids_vectors.p'
    with open(ids_vectors_path, 'rb') as f:
        ids_vectors = pickle.load(f)

    def id_to_vector(id_):
        try:
            return ids_vectors[id_]
        except:
            pass

    return id_to_vector

UPDATE_FAISS_AFTER_SECONDS = None

INDEX_INPUT_QUEUE = None

INDEX_DIMENSIONS = None

IDS_MAP_FILE_PATH = None

INDEX_FILE_PATH = None

FASTTEXT_MODEL_PATH = None
