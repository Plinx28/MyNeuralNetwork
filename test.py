import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from config import DATA_DIR, IMAGE_SIZE
import matplotlib.pyplot as plot

test_data_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_data_gen.flow_from_directory(
        directory=DATA_DIR + '/test',
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode='categorical')

model = load_model('model.h5')

test_loss, test_accurancy = model.evaluate_generator(generator=test_generator, steps=test_generator.n // test_generator.batch_size)

print(f'Test loss:{test_loss}')
print(f'Test accuracy:{test_accurancy}')

# for i in range(len(test_generator)):
#     image = test_generator[i]ve
    
#     plot.imshow(image)
#     plot.axis('off')
        
#     plot.show()
