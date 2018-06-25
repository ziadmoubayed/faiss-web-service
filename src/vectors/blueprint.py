from flask import Blueprint
from src.vectors.vector_utils import VectorUtils

blueprint = Blueprint('vector_utils', __name__)


@blueprint.record_once
def setup_vectors(setup_state):
    model_path = setup_state.app.config.get('FASTTEXT_MODEL_PATH')
    language = setup_state.app.config.get('LANGUAGE')
    VectorUtils.init(model_path, language)
