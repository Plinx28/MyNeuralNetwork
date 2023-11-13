import tensorflow as tf
import os

tf.config.set_visible_devices([], 'GPU')

DATA_DIR = 'data'

TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

IMAGE_SIZE = (150, 150)

NUM_CLASSES = 5