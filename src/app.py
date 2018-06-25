from flask import Flask

from src.internal.blueprint import blueprint as InternalBlueprint
from src.faiss_index.blueprint import blueprint as FaissIndexBlueprint

app = Flask(__name__)
app.config.from_object('config')
app.config.from_pyfile('/home/gorih/PycharmProjects/faiss-web-service/resources/faiss_index_local_file.py')
# app.config.from_pyfile('/home/gorih/PycharmProjects/faiss-web-service/resources/fais_index_generated.py')
# app.config.from_envvar('FAISS_WEB_SERVICE_CONFIG')

app.register_blueprint(InternalBlueprint)
app.register_blueprint(FaissIndexBlueprint)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
