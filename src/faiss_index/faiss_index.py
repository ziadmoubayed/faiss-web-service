import numpy as np
import redis


class FaissIndex(object):
    def __init__(self, index, id_to_vector):
        assert index
        assert id_to_vector

        self.index = index
        self.id_to_vector = id_to_vector
        self.redis = redis.Redis()

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
        return { 'id': uuid, 'score': float(score), 'text': self.redis.get("uuid_vs_body:"+uuid).decode("utf-8")}

    def __search__(self, ids, vectors, k):

        def result_dict(id_, vector, neighbors):
            return { 'id': id_, 'vector': vector.tolist(), 'neighbors': neighbors }

        results = []

        vectors = [np.array(vector, dtype=np.float32) for vector in vectors]
        vectors = np.atleast_2d(vectors)

        scores, neighbors = self.index.search(vectors, k) if vectors.size > 0 else ([], [])
        for id_, vector, neighbors, scores in zip(ids, vectors, neighbors, scores):
            neighbors_scores = zip(neighbors, scores)
            neighbors_scores = [(n, s) for n, s in neighbors_scores if n != id_ and n != -1]
            neighbors_scores = [self.neighbor_dict(n, s) for n, s in neighbors_scores]

            results.append(result_dict(id_, vector, neighbors_scores))

        return results
