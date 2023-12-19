from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

import matplotlib.pyplot as plot

from config import IMAGE_SIZE, DATA_DIR
from model import model


def main():
    train_generator = build_train_generator()
    val_generator = build_val_generator()

    optimazer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimazer, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint("model.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                  save_weights_only=False, mode='auto')
    
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=0.99, mode='auto')

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=16,
        validation_data=val_generator,
        validation_steps=val_generator.n // val_generator.batch_size,
        callbacks=[checkpoint, early])

    plot.plot(history.history['accuracy'], linestyle='dotted')
    plot.plot(history.history['val_accuracy'])
    plot.grid(True)
    plot.show()


def build_train_generator():
    train_data_gen = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_gen.flow_from_directory(
        directory=DATA_DIR + '/train',
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale')
    
    return train_generator


def build_val_generator():
    val_data_gen = ImageDataGenerator(rescale=1./255)
    
    val_generator = val_data_gen.flow_from_directory(
        directory=DATA_DIR + '/val',
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale')
    
    return val_generator


if __name__ == '__main__':
    main()