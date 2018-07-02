from flask import Flask

from internal.blueprint import blueprint as InternalBlueprint
from faiss_index.blueprint import blueprint as FaissIndexBlueprint
from vectors.vector_utils import VectorUtils

app = Flask(__name__)
# app.config.from_object('config')
app.config.from_pyfile('../resources/configurations.py')

app.register_blueprint(InternalBlueprint)
app.register_blueprint(FaissIndexBlueprint)


def initiate_application(app):
    # Loads and initiates fasttext's model for transforming documents to vectors
    model_path = app.config.get('WORDS_VECTORS_FILE_PATH')
    VectorUtils.language = app.config.get('LANGUAGE')
    should_load_vocabulary = app.config.get('LOAD_VOCABULARY')

    if should_load_vocabulary:
        VectorUtils.load_vocabulary(model_path)

if __name__ == "__main__":
    initiate_application(app)
    app.run(app.config.get("APP_HOST"), app.config.get('APP_PORT'))
