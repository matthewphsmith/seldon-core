from YOUR_LOADING_LIBRARY import YOUR_LOADING_FUNCTION


class ModelName(object):
    def __init__(self, YOUR_MODEL_FILE):
        self.model = YOUR_LOADING_LIBRARY(YOUR_MODEL_FILE)

    def predict(self, X, features_names):
        return self.model.predict(X)
