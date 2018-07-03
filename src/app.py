from flask import Flask

from internal.blueprint import blueprint as InternalBlueprint
from faiss_index.blueprint import blueprint as FaissIndexBlueprint
from vocabulary.vocabulary import VocabularyKeeper

app = Flask(__name__)
app.config.from_pyfile('../resources/configurations.py')

app.register_blueprint(InternalBlueprint)
app.register_blueprint(FaissIndexBlueprint)


def initiate_application(app):
    # Loads and initiates fasttext's model for transforming documents to vectors
    model_path = app.config.get('WORDS_VECTORS_FILE_PATH')
    should_load_vocabulary = app.config.get('LOAD_VOCABULARY')
    vocabulary_in_memory = app.config.get('VOCABULARY_IN_MEMORY')
    redis_host = app.config.get('REDIS_HOST')
    redis_port = app.config.get('REDIS_PORT')
    redis_db = app.config.get('REDIS_DB')
    VocabularyKeeper.init(should_load_vocabulary, vocabulary_in_memory, model_path, redis_host, redis_port, redis_db)

initiate_application(app)

if __name__ == "__main__":
    app.run(app.config.get("APP_HOST"), app.config.get('APP_PORT'))
