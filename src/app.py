from flask import Flask

from internal.blueprint import blueprint as InternalBlueprint
from faiss_index.blueprint import blueprint as FaissIndexBlueprint
from vocabulary.vocabulary import VocabularyKeeper
from vocabulary.doc2vec_model import Doc2VecModelKeeper
from vocabulary.infer_sent import InferSentModelKeeper

def initiate_application(app):
    # Loads and initiates fasttext's model for transforming documents to vectors
    vector_utils_type = app.config.get('VECTOR_UTILS_TYPE')
    model_path = app.config.get('WORDS_VECTORS_FILE_PATH')

    if vector_utils_type == 'inferSent':
        InferSentModelKeeper.init(app.config.get('INFERSENT_ENCODER_FILE_PATH'), model_path, app.config.get('INFERSENT_WORDS_VOCABULARY_INITIAL_SIZE'))
    elif vector_utils_type == 'average':
        should_load_vocabulary = app.config.get('LOAD_VOCABULARY')
        vocabulary_in_memory = app.config.get('VOCABULARY_IN_MEMORY')
        redis_host = app.config.get('REDIS_HOST')
        redis_port = app.config.get('REDIS_PORT')
        redis_db = app.config.get('REDIS_DB')
        VocabularyKeeper.init(should_load_vocabulary, vocabulary_in_memory, model_path, redis_host, redis_port, redis_db)
    elif vector_utils_type == 'sentence':
        load_doc2vec_model = app.config.get('LOAD_DOC2VEC_MODEL')
        doc2vec_model_path = app.config.get('DOC2VEC_MODEL_FILE_PATH')
        if load_doc2vec_model:
            Doc2VecModelKeeper.init(doc2vec_model_path)



app = Flask(__name__)
app.config.from_pyfile('../resources/configurations.py')

initiate_application(app)


app.register_blueprint(InternalBlueprint)
app.register_blueprint(FaissIndexBlueprint)

if __name__ == "__main__":
    app.run(app.config.get("APP_HOST"), app.config.get('APP_PORT'))
