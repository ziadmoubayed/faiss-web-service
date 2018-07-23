from jsonschema import validate, ValidationError
from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest
from faiss_index.faiss_index import FaissIndex
# from vectors.infersent_vector_utils import VectorUtils # u can use different versions of VectorUtils from vectors package
import json
import logging as log
import sys

log.basicConfig(stream=sys.stdout, level=log.DEBUG)

blueprint = Blueprint('faiss_index', __name__)

@blueprint.record_once
def record(setup_state):
    redis_host = setup_state.app.config.get('REDIS_HOST')
    redis_port = setup_state.app.config.get('REDIS_PORT')
    redis_db = setup_state.app.config.get('REDIS_DB')
    vec_ut_type = setup_state.app.config.get('VECTOR_UTILS_TYPE')
    index_dimensions = setup_state.app.config.get('INDEX_DIMENSIONS')

    if vec_ut_type == 'inferSent':
        from vectors.infersent_vector_utils import VectorUtils
        vector_utills = VectorUtils(setup_state.app.config.get('LANGUAGE'), redis_host, redis_port, redis_db)
        index_dimensions = 4096
    elif vec_ut_type == 'average':
        from vectors.average_vector_utils import VectorUtils
        vector_utills = VectorUtils(setup_state.app.config.get('LANGUAGE'), redis_host, redis_port, redis_db)
    elif vec_ut_type == "averageStem":
        from vectors.average_stem_vector_utils import VectorUtils
        vector_utills = VectorUtils(setup_state.app.config.get('LANGUAGE'), redis_host, redis_port, redis_db)
    elif vec_ut_type == 'sentence':
        from vectors.sentence_vector_utils import VectorUtils
        vector_utills = VectorUtils(setup_state.app.config.get('LANGUAGE'), redis_host, redis_port, redis_db)
    else:
        from vectors.sentence_stem_vector_utils import VectorUtils
        vector_utills = VectorUtils(setup_state.app.config.get('LANGUAGE'), redis_host, redis_port, redis_db)

    manage_faiss_index(
        index_dimensions,
        setup_state.app.config.get('INDEX_INPUT_QUEUE'),
        setup_state.app.config.get('INDEX_FILE_PATH'),
        setup_state.app.config.get('IDS_MAP_FILE_PATH'),
        redis_host, redis_port, redis_db, vector_utills,
        setup_state.app.config.get('INDEX_WRITES_FREQUENCY_SEC'),
        setup_state.app.config.get('PERSIST_BODIES'))

@blueprint.route('/vector', methods=['GET'])
def get_vector():
    body = request.args.get('body')
    return json.dumps(blueprint.vector_utils.getVector(body).tolist())


@blueprint.route('/faiss/similar', methods=['GET'])
def get_similar():
    import numpy as np
    import time
    body = request.args.get('body')
    limit = request.args.get('limit')
    start_time = int(round(time.time() * 1000))
    vector = np.array(blueprint.vector_utils.getVector(body))
    vectors = [vector]
    time_took = int(round(time.time() * 1000)) - start_time
    log.info("Text was embedded in %d millis ", time_took)
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
        log.error('Bad request', e)
        return 'Bad request', 400

    except Exception as e:
        log.error('Server error', e)
        return 'Server error', 500

def manage_faiss_index(d, input_queeu, faiss_index_path, ids_mapping_path, redis_host, redis_port, redis_db, vector_utils, save_index_frequency, should_persist_bodies):
    blueprint.faiss_index = FaissIndex(d, input_queeu, faiss_index_path, ids_mapping_path, redis_host, redis_port, redis_db, save_index_frequency, should_persist_bodies, vector_utils)
    blueprint.vector_utils = vector_utils

