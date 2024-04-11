# import tensorflow as tf
import os

# Turn off CUDA
# tf.config.set_visible_devices([], 'GPU')

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

IMAGE_SIZE = (256, 256)

NUM_CLASSES = len(os.listdir(os.path.join(DATA_DIR, 'train')))

BATCH_SIZE = 128

COLOR_MODE = 'grayscale'

LEARNING_RATE = 0.001
# model of trained model
MODEL_NAME = 'model.h5'
PRETRAINED_MODEL = 'model_9160.h5'

# pip3 install --upgrade tensorflow==2.10.0

