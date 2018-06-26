from flask import Flask

from src.internal.blueprint import blueprint as InternalBlueprint
from src.faiss_index.blueprint import blueprint as FaissIndexBlueprint
from src.vectors.vector_utils import VectorUtils

app = Flask(__name__)
app.config.from_object('config')
app.config.from_pyfile('/home/gorih/PycharmProjects/faiss-web-service/resources/faiss_index_local_file.py')

app.register_blueprint(InternalBlueprint)
app.register_blueprint(FaissIndexBlueprint)


def initiate_application(app):
    print("Initiating protocol launched")

    # Loads and initiates fasttext's model for transforming documents to vectors
    model_path = app.config.get('FASTTEXT_MODEL_PATH')
    language = app.config.get('LANGUAGE')
    VectorUtils.init(model_path, language)

if __name__ == "__main__":
    initiate_application(app)
    app.run(host='0.0.0.0')
