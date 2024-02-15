import tensorflow as tf
import os

# Turn off CUDA
tf.config.set_visible_devices([], 'GPU')

DATA_DIR = 'data'
"""
This is main data-directory with structure:
    test/
        class_1/
        class_2/
        ...
    train/
        class_1/
        class_2/
        ...
    val/
        class_1/
        class_2/
        ...
"""


TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

IMAGE_SIZE = (256, 256)

NUM_CLASSES = 3

BATCH_SIZE = 32

COLOR_MODE = 'grayscale'

# model of trained model
MODEL_NAME = 'model_9160.h5'

