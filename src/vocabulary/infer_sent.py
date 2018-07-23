import torch
from vocabulary.infer_sent_src.models import InferSent

class InferSentModelKeeper():
    model = None

    @staticmethod
    def init(model_path, word2vec_file_path):
        def vocab_from_words(model):
            model.build_vocab_k_words(K=100000)

        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}

        InferSentModelKeeper.model = InferSent(params_model)
        InferSentModelKeeper.model.load_state_dict(torch.load(model_path))
        InferSentModelKeeper.model.set_w2v_path(word2vec_file_path)

        vocab_from_words(InferSentModelKeeper.model)
