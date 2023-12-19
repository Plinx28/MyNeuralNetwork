from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from config import NUM_CLASSES, IMAGE_SIZE


"""Model with 91.60% validation accuracy"""
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2, 
                  input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

