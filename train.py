import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from dotenv import load_dotenv
import matplotlib.pyplot as plot

from config import IMAGE_SIZE, DATA_DIR
from model import model

def main():
    train_data_gen = ImageDataGenerator(rescale=1./255)
    val_data_gen = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_gen.flow_from_directory(
        directory=DATA_DIR + '/train',
        target_size=IMAGE_SIZE,
        batch_size=16,
        class_mode='categorical')

    val_generator = val_data_gen.flow_from_directory(
        directory=DATA_DIR + '/val',
        target_size=IMAGE_SIZE,
        batch_size=16,
        class_mode='categorical')

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=20,
        validation_data=val_generator,
        validation_steps=val_generator.n // val_generator.batch_size)

    model.save('model.h5')

    plot.plot(history.history['loss'])
    plot.plot(history.history['val_loss'])
    plot.grid(True)
    plot.show()


if __name__ == '__main__':
    main()