import logging as log
import sys
import time

log.basicConfig(stream=sys.stdout, level=log.DEBUG)

class VocabularyKeeper():
    vocabulary = None
    in_memory = False

    @staticmethod
    def init(load_vocabulary, load_to_memory, model_path, redis_host='localhost', redis_port=6379, redis_db=0):
        VocabularyKeeper.in_memory = load_to_memory

        def load_vocabulary_to_redis(vectors_model_path):
            log.info("Started loading of vocabulary to redis.")
            import redis
            redis = redis.Redis(redis_host, redis_port, redis_db)
            with open(vectors_model_path) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    tokens = line.rstrip().split(' ')
                    word = tokens[0]
                    log.debug("Saving word ' %s ' to vocabulary. Loaded %d" % (word, cnt))
                    redis.hset('vocabulary', word, tokens[1:])
                    cnt += 1

        def load_vocabulary_to_memory(vectors_model_path):
            log.info("Started loading of vocabulary to memory.")
            vocabulary = {}
            with open(vectors_model_path) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    tokens = line.rstrip().split(' ')
                    word = tokens[0]
                    log.debug("Saving word ' %s ' to vocabulary. Loaded %d" % (word, cnt))
                    vocabulary[word] = map(float, tokens[1:])
                    cnt += 1

            return vocabulary

        start = time.time()

        if load_vocabulary:
            if load_to_memory:
                VocabularyKeeper.vocabulary = load_vocabulary_to_memory(model_path)
            else:
                load_vocabulary_to_redis(model_path)

        log.info('Loading Vectors took %0.3f ms' % ((time.time() - start) * 1000.0))


