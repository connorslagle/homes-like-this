import pandas as pd
# from .json_pipeline import Cleaner
import pickle

class Predictor():

    def __init__(self, model_file):
        model_file = f'models/{model_file}'
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

    def fit_new_json(self, json_file):
        self.y_proba = self.model.predict_proba(json_file)
        


if __name__ == "__main__":
    'hi'