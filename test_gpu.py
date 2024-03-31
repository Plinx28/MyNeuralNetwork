import tensorflow as tf

print(f'Tensorflow version: {tf.__version__}')
print(f'Available GPUs: {tf.test.gpu_device_name()}')
print(f'GPU is available: {tf.config.list_physical_devices("GPU")}')
