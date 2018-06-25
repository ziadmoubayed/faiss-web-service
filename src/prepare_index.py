import json
import pickle
import time
from threading import Thread, Condition

import faiss
import numpy as np
import redis

from src.vectors.vector_utils import VectorUtils

d = 50
index_input_queue = "doc2vector"
path_for_index = "/home/gorih/PycharmProjects/faiss-web-service/resources/index"
id_to_uuid_file_path = "/home/gorih/PycharmProjects/faiss-web-service/resources/ids_vectors.p"
id_vs_uuid = {}

index = faiss.IndexFlatL2(d)  # build the index
xb = np.zeros(shape=(0, d)).astype('float32')
index.add(xb)
c = Condition()
vectors = VectorUtils.init("/home/gorih/Documents/fastText/klangoo-rus.bin", "russian")
indexSize = 0


def saveIndex():
    global index
    global id_vs_uuid
    print("Saving the index")
    faiss.write_index(index, path_for_index)
    with open(id_to_uuid_file_path, 'wb') as handle:
        pickle.dump(id_vs_uuid, handle, protocol=pickle.HIGHEST_PROTOCOL)


def addToIndex(uuid, vector):
    c.acquire()
    global indexSize
    global index
    print("Index Size is ", index.ntotal)
    print("Setting uuid at ", indexSize)
    id_vs_uuid[indexSize] = uuid
    indexSize += 1
    xb2 = np.zeros(shape=(1, d)).astype('float32')
    xb2[0] = vector
    index.add(xb2)
    c.release()


def run():
    vectors = VectorUtils()
    r = redis.Redis()

    def time_out(last_update):
        return 5 < time.time() - last_update

    last_index_update = 0
    while True:
        value = r.lpop(index_input_queue)
        while value and not time_out(last_index_update):
            data = json.loads(value)
            value_str = data['body']
            uuid = data['uuid']
            vector = vectors.getVector(value_str)
            addToIndex(uuid, vector)
            r.set("uuid_vs_body:" + uuid, value_str)
            value = r.lpop(index_input_queue)

        saveIndex()
        time.sleep(10)
        last_index_update = time.time()


from flask import Flask

app = Flask(__name__)

@app.route("/size")
def size():
    global index
    return index.ntotal


def generate_index():
    thread = Thread(target=run)
    thread.start()

if __name__ == '__main__':
    generate_index()
