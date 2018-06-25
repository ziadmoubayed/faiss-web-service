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
        setup_state.app.config.get('GET_FAISS_RESOURCES'),
        setup_state.app.config['GET_FAISS_INDEX'],
        setup_state.app.config['GET_FAISS_ID_TO_VECTOR'],
        setup_state.app.config.get('UPDATE_FAISS_AFTER_SECONDS'))

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
        json_input = request.get_json(force=True)
        validate(json_input, {
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

def manage_faiss_index(get_faiss_resources, get_faiss_index, get_faiss_id_to_vector, update_after_seconds):

    SIGNAL_SET_FAISS_RESOURCES = 1
    SIGNAL_SET_FAISS_INDEX = 2

    def set_faiss_resources(signal = None):
        print('Getting Faiss resources')
        get_faiss_resources()

        if uwsgi and signal:
            uwsgi.signal(SIGNAL_SET_FAISS_INDEX)

    def set_faiss_index(signal = None):
        print('Getting Faiss index')
        blueprint.faiss_index = FaissIndex(get_faiss_index(), get_faiss_id_to_vector())

    def set_periodically():
        if isinstance(update_after_seconds, int):

            uwsgi.register_signal(SIGNAL_SET_FAISS_INDEX, 'workers', set_faiss_index)

            if get_faiss_resources:
                uwsgi.register_signal(SIGNAL_SET_FAISS_RESOURCES, 'worker', set_faiss_resources)
                uwsgi.add_timer(SIGNAL_SET_FAISS_RESOURCES, update_after_seconds)
            else:
                uwsgi.add_timer(SIGNAL_SET_FAISS_INDEX, update_after_seconds)

        else:
            print('Failed to set periodic faiss index updates')
            print('UPDATE_FAISS_AFTER_SECONDS must be an integer')

    if uwsgi and update_after_seconds:
        set_periodically()

    if get_faiss_resources:
        set_faiss_resources()

    set_faiss_index()
