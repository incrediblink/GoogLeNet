from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
import matplotlib.pyplot as plt

layers = tf.keras.layers
models = tf.keras.models
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

batch_size = 128
epochs = 10
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

def getInception(one, threeReduce, three, fiveReduce, five, pool, input):
  inc_conv_1x1_1_0 = layers.Conv2D(one, (1, 1), activation='relu')(input)

  inc_conv_1x1_1_1 = layers.Conv2D(threeReduce, (1, 1), activation='relu')(input)
  inc_conv_3x3_1_1 = layers.Conv2D(three, (3, 3), padding='same', activation='relu')(inc_conv_1x1_1_1)

  inc_conv_1x1_1_2 = layers.Conv2D(fiveReduce, (1, 1), activation='relu')(input)
  inc_conv_5x5_1_2 = layers.Conv2D(five, (5, 5), padding='same', activation='relu')(inc_conv_1x1_1_2)

  inc_max_3x3_1_3 = layers.MaxPooling2D((3, 3), padding='same', strides=1)(input)
  inc_conv_1x1_1_3 = layers.Conv2D(pool, (1, 1), activation='relu')(inc_max_3x3_1_3)

  return layers.concatenate([
    inc_conv_1x1_1_0,
    inc_conv_3x3_1_1,
    inc_conv_5x5_1_2,
    inc_conv_1x1_1_3
  ])

normalizationEpsilon = 1e-6

input = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
conv_7x7_2_1 = layers.Conv2D(64, (7, 7), activation='relu', strides=2, padding='same')(input)
max_3x3_2_1 = layers.MaxPooling2D((3, 3), strides=2, padding='same')(conv_7x7_2_1)
norm_1 = layers.LayerNormalization(epsilon=normalizationEpsilon)(max_3x3_2_1)

conv_1x1_1_2 = layers.Conv2D(64, (1, 1), activation='relu')(norm_1)
conv_3x3_1_2 = layers.Conv2D(192, (3, 3), activation='relu', padding='same')(conv_1x1_1_2)
norm_2 = layers.LayerNormalization(epsilon=normalizationEpsilon)(conv_3x3_1_2)
max_3x3_2_2 = layers.MaxPooling2D((3, 3), strides=2, padding='same')(norm_2)

inc_3a = getInception(64, 96, 128, 16, 32, 32, max_3x3_2_2)
inc_3b = getInception(128, 128, 192, 32, 96, 64, inc_3a)
max_3x3_2_3 = layers.MaxPooling2D((3, 3), strides=2, padding='same')(inc_3b)

inc_4a = getInception(192, 96, 208, 16, 48, 64, max_3x3_2_3)
inc_4b = getInception(160, 112, 224, 24, 64, 64, inc_4a)
inc_4c = getInception(128, 128, 256, 24, 64, 64, inc_4b)
inc_4d = getInception(112, 144, 288, 32, 64, 64, inc_4c)
inc_4e = getInception(256, 160, 320, 32, 128, 128, inc_4d)
max_3x3_2_4 = layers.MaxPooling2D((3, 3), strides=2, padding='same')(inc_4e)

inc_5a = getInception(256, 160, 320, 32, 128, 128, max_3x3_2_4)
inc_5b = getInception(384, 192, 384, 48, 128, 128, inc_5a)

avg_6 = layers.AveragePooling2D((7, 7), padding='same')(inc_5b)
dropout_6 = layers.Dropout(0.4)(max_3x3_2_1)
flatten = layers.Flatten()(dropout_6)
fc_6 = layers.Dense(1000, activation='relu')(flatten)
fc_7 = layers.Dense(1, activation='sigmoid')(fc_6)


model = tf.keras.Model(inputs=input, outputs=fc_7)
model.summary()

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
