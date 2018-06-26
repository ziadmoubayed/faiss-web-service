import faiss
import numpy as np
import time
import pickle
import redis
import json

from threading import Thread, Condition
from src.vectors.vector_utils import VectorUtils


class FaissIndex(object):
    def __init__(self, d, index_input_queue, path_for_index, id_to_uuid_file_path, redis_host, redis_port, redis_db,
                 save_index_frequency):
        print("Instantiating of FaissIndex")
        self.d = d
        self.index_input_queue = index_input_queue
        self.path_for_index = path_for_index
        self.id_to_uuid_file_path = id_to_uuid_file_path
        self.lock = Condition()
        self.index_size = 0
        self.last_index_update = 0
        self.save_index_frequency = save_index_frequency

        try:
            self.index = faiss.read_index(path_for_index)
            self.index_size = self.index.ntotal
        except:
            self.index = faiss.IndexFlatL2(d)
            xb = np.zeros(shape=(0, d)).astype('float32')
            self.index.add(xb)

        try:
            with open(id_to_uuid_file_path, 'rb') as f:
                self.ids_mapping = pickle.load(f)
        except:
            self.ids_mapping = {}

        self.vectros = VectorUtils()
        self.redis = redis.Redis(redis_host, redis_port, redis_db)
        self.generate_index()

    def id_to_vector(self, id_):
        return self.ids_mapping[id_]

    def search_by_ids(self, ids, k):
        vectors = [self.id_to_vector(id_) for id_ in ids]
        results = self.__search__(ids, vectors, k + 1)

        return results

    def search_by_vectors(self, vectors, k):
        ids = [None] * len(vectors)
        results = self.__search__(ids, vectors, k)

        return results

    def neighbor_dict(self, id_, score):
        uuid = self.id_to_vector(int(id_))
        return {'id': uuid, 'score': float(score), 'text': self.redis.get("uuid_vs_body:" + uuid).decode("utf-8")}

    def __search__(self, ids, vectors, k):
        self.lock.acquire()

        def result_dict(id_, vector, neighbors):
            return {'id': id_, 'vector': vector.tolist(), 'neighbors': neighbors}

        results = []

        vectors = [np.array(vector, dtype=np.float32) for vector in vectors]
        vectors = np.atleast_2d(vectors)

        scores, neighbors = self.index.search(vectors, k) if vectors.size > 0 else ([], [])
        for id_, vector, neighbors, scores in zip(ids, vectors, neighbors, scores):
            neighbors_scores = zip(neighbors, scores)
            neighbors_scores = [(n, s) for n, s in neighbors_scores if n != id_ and n != -1]
            neighbors_scores = [self.neighbor_dict(n, s) for n, s in neighbors_scores]

            results.append(result_dict(id_, vector, neighbors_scores))
        self.lock.release()
        return results

    ########################

    def saveIndex(self):
        print("Saving the index")
        faiss.write_index(self.index, self.path_for_index)
        with open(self.id_to_uuid_file_path, 'wb') as handle:
            pickle.dump(self.ids_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _addToIndex(self, uuid, vector):
        self.lock.acquire()
        print("Index Size is ", self.index.ntotal)
        print("Setting uuid at ", self.index_size)
        self.ids_mapping[self.index_size] = uuid
        self.index_size += 1
        xb2 = np.zeros(shape=(1, self.d)).astype('float32')
        xb2[0] = vector
        self.index.add(xb2)
        self.lock.release()

    def time_out(self):
        current_time = time.time()
        time_pass = current_time - self.last_index_update
        self.last_index_update = current_time
        print("Time pass : " + str(time_pass))
        return self.save_index_frequency < time_pass

    def run(self):
        vectors = VectorUtils()

        while True:
            self.get_new_vectors(vectors)

    def get_new_vectors(self, v):
        index_size_before_update = self.index.ntotal
        new_vectors_counter = index_size_before_update

        value = self.redis.lpop(self.index_input_queue)
        while value and not self.time_out():
            data = json.loads(value)
            value_str = data['body']
            uuid = data['uuid']
            vector = v.getVector(value_str)
            self._addToIndex(uuid, vector)
            new_vectors_counter += 1
            self.redis.set("uuid_vs_body:" + uuid, value_str)
            value = self.redis.lpop(self.index_input_queue)

        if (index_size_before_update < new_vectors_counter):
            self.saveIndex()

    def generate_index(self):
        print("Launching the thread")
        thread = Thread(target=self.run)
        thread.start()
