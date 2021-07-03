from pathlib import Path
import PIL.Image as PImage
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from collections import Counter

class CNN_Model:
    def __init__(self):
        self.configSet = False

    def create_ds(self, directory):

        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           vertical_flip=False,
                                           horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_gen = train_datagen.flow_from_directory('{}/train'.format(directory),
                                                      target_size=(150, 150),
                                                      batch_size=32,
                                                      class_mode='categorical')


        self.val_gen = test_datagen.flow_from_directory('{}/test'.format(directory),
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='categorical')

        self.classes = list(self.train_gen.class_indices.keys())
        self.train_num = Counter(self.train_gen.classes)
        self.test_num = Counter(self.val_gen.classes)

        if not self.configSet:
            self.model_config()
        else:
            self.configSet = True

    def model_config(self):
        # Initialising the CNN
        self.cnn = tf.keras.models.Sequential()

        # Step 1 - Convolution
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[150, 150, 3]))

        # Step 2 - Pooling
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Adding convolutional layer
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Step 3 - Flattening
        self.cnn.add(tf.keras.layers.Flatten())

        # Step 4 - Full Connection
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

        # Step 5 - Output Layer
        self.cnn.add(tf.keras.layers.Dense(units=self.train_gen.num_classes, activation='softmax'))

        # Loading model weights
        if Path('model.h5').exists():
            self.cnn.load_weights('model.h5')

        # Compiling the CNN
        self.cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        self.info = self.cnn.fit(self.train_gen, validation_data=self.val_gen, epochs=1,
                                 workers=4)

    def save_weights(self):
        self.cnn.save_weights('model.h5')

    def predict_img(self, img):
        img_tensor = img.resize((150, 150), PImage.ANTIALIAS)
        img_tensor = img_to_array(img_tensor)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        prediction = self.cnn.predict(img_tensor)
        return self.classes[np.argmax(prediction)], 100 * np.max(prediction)
