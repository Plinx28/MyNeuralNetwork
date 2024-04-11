from keras.models import load_model
from config import MODEL_NAME

import os
from config import DATA_DIR
from train import build_generator


def test_model(model_name):
    model_loaded = load_model(model_name)
    accuracy = model_loaded.evaluate(build_generator("test"), verbose=1)

    print(f'Точность на тестовой выборке: {accuracy[1] * 100}%')

    return accuracy


test_model(MODEL_NAME)
