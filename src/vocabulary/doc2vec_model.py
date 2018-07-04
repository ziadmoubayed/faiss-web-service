import fastText as fast


class Doc2VecModelKeeper():
    model = None

    @staticmethod
    def init(model_path):
        Doc2VecModelKeeper.model = fast.load_model(model_path)
