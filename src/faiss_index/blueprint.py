from jsonschema import validate, ValidationError
from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest
from src.faiss_index.faiss_index import FaissIndex
from src.vectors.vector_utils import VectorUtils
import json

try:
    import uwsgi
except ImportError:
    print('Failed to load python module uwsgi')
    print('Periodic faiss index updates isn\'t enabled')

    uwsgi = None

blueprint = Blueprint('faiss_index', __name__)

@blueprint.record_once
def record(setup_state):
    manage_faiss_index(
        setup_state.app.config.get('INDEX_DIMENSIONS'),
        setup_state.app.config.get('INDEX_INPUT_QUEUE'),
        setup_state.app.config.get('INDEX_FILE_PATH'),
        setup_state.app.config.get('IDS_MAP_FILE_PATH'),
        setup_state.app.config.get('REDIS_HOST'),
        setup_state.app.config.get('REDIS_PORT'),
        setup_state.app.config.get('REDIS_DB'),
        setup_state.app.config.get('INDEX_WRITES_FREQUENCY_SEC'))

@blueprint.route('/vector', methods=['GET'])
def get_vector():
    body = request.args.get('body')
    return json.dumps(VectorUtils().getVector(body).tolist())


@blueprint.route('/faiss/similar', methods=['GET'])
def get_similar():
    import numpy as np
    body = request.args.get('body')
    limit = request.args.get('limit')
    vec_utils = VectorUtils()
    vector = np.array(vec_utils.getVector(body))
    vectors = [vector]
    results_vectors = blueprint.faiss_index.search_by_vectors(vectors, int(limit))
    return jsonify(results_vectors)


@blueprint.route('/faiss/search', methods=['POST'])
def search():
    try:
        json = request.get_json(force=True)
        validate(json, {
            'type': 'object',
            'required': ['k'],
            'properties': {
                'k': { 'type': 'integer', 'minimum': 1 },
                'ids': { 'type': 'array', 'items': { 'type': 'number' }},
                'vectors': {
                    'type': 'array',
                    'items': {
                        'type': 'array',
                        'items': { 'type': 'number' }
                    }
                }
            }
        })

        results_ids = blueprint.faiss_index.search_by_ids(json['ids'], json['k']) if 'ids' in json else []
        results_vectors = blueprint.faiss_index.search_by_vectors(json['vectors'], json['k']) if 'vectors' in json else []

        return jsonify(results_ids + results_vectors)

    except (BadRequest, ValidationError) as e:
        print('Bad request', e)
        return 'Bad request', 400

    except Exception as e:
        print('Server error', e)
        return 'Server error', 500

def manage_faiss_index(d, input_queeu, faiss_index_path, ids_mapping_path, redis_host, redis_port, redis_db, save_index_frequency):
    blueprint.faiss_index = FaissIndex(d, input_queeu, faiss_index_path, ids_mapping_path, redis_host, redis_port, redis_db, save_index_frequency)

