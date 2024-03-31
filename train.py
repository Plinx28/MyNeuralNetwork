import os
from typing import Literal
from datetime import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

import matplotlib.pyplot as plot

from config import IMAGE_SIZE, DATA_DIR, BATCH_SIZE, COLOR_MODE, MODEL_NAME
from model import model


Subsample = Literal['train', 'val', 'test']


def build_generator(subsample: Subsample):
    """Create generator"""
    data_gen = ImageDataGenerator(rescale=1./255)
    generator = data_gen.flow_from_directory(
        directory=os.path.join(DATA_DIR, subsample),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode=COLOR_MODE)

    return generator


def main():
    train_generator = build_generator('train')
    val_generator = build_generator('val')

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint("model.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                        save_weights_only=False, mode='auto'),
        EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
        ]


    history = model.fit(
        x=train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=22,
        validation_data=val_generator,
        validation_steps=val_generator.n // val_generator.batch_size,
        callbacks=callbacks
    )

    save_time = datetime.now()
    model_name_now = f"{MODEL_NAME}_{save_time.date()}_{save_time.hour}-{save_time.minute}-{save_time.second}.h5"
    model.save(model_name_now)
    print(f"Модель была сохранена в файле {model_name_now}")

    plot.plot(history.history['accuracy'], linestyle='dotted')
    plot.plot(history.history['val_accuracy'])
    plot.grid(True)
    plot.show()


if __name__ == '__main__':
    main()
