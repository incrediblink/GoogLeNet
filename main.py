from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

layers = tf.keras.layers
datasets = tf.keras.datasets
models = tf.keras.models

dataset = tfds.load(name='imagenet2012', split=tfds.Split.TRAIN)

(train_images, train_labels), (test_images, test_labels) = dataset

train_images, test_images = train_images / 255.0, test_images / 255.0

def getInception(one, threeReduce, three, fiveReduce, five, pool, input):
  inc_conv_1x1_1_0 = layers.Conv2D(one, (1, 1))(input)

  inc_conv_1x1_1_1 = layers.Conv2D(threeReduce, (1, 1))(input)
  inc_conv_3x3_1_1 = layers.Conv2D(three, (3, 3))(inc_conv_1x1_1_1)

  inc_conv_1x1_1_2 = layers.Conv2D(fiveReduce, (1, 1))(input)
  inc_conv_5x5_1_2 = layers.Conv2D(five, (5, 5))(inc_conv_1x1_1_2)

  inc_max_3x3_1_3 = layers.MaxPooling2D((3, 3))(input)
  inc_conv_1x1_1_3 = layers.Conv2D(pool, (1, 1))(inc_max_3x3_1_3)

  return layers.concatenate([
    inc_conv_1x1_1_0,
    inc_conv_3x3_1_1,
    inc_conv_5x5_1_2,
    inc_conv_1x1_1_3
  ])

normalizationEpsilon = 1e-6

input = layers.Input(shape=(32, 32, 3))
conv_7x7_2_1 = layers.Conv2D(32, (7, 7), activation='relu', input_shape=(32, 32, 3), strides=2)(input)
max_3x3_2_1 = layers.MaxPooling2D((3, 3), strides=2)(conv_7x7_2_1)
norm_1 = layers.LayerNormalization(epsilon=normalizationEpsilon)(max_3x3_2_1)

conv_1x1_1_2 = layers.Conv2D(64, (1, 1), activation='relu')(norm_1)
conv_3x3_1_2 = layers.Conv2D(192, (3, 3), activation='relu')(conv_1x1_1_2)
norm_2 = layers.LayerNormalization(epsilon=normalizationEpsilon)(conv_3x3_1_2)
max_3x3_2_2 = layers.MaxPooling2D((3, 3), strides=2)(norm_2)

inc_3a = getInception(64, 96, 128, 16, 32, 32, max_3x3_2_2)
inc_3b = getInception(128, 128, 192, 32, 96, 64, inc_3a)
max_3x3_2_3 = layers.MaxPooling2D((3, 3), strides=2)(inc_3b)

inc_4a = getInception(192, 96, 208, 16, 48, 64, max_3x3_2_3)
inc_4b = getInception(160, 112, 224, 24, 64, 64, inc_4a)
inc_4c = getInception(128, 128, 256, 24, 64, 64, inc_4b)
inc_4d = getInception(112, 144, 288, 32, 64, 64, inc_4c)
inc_4e = getInception(256, 160, 320, 32, 128, 128, inc_4d)
max_3x3_2_4 = layers.MaxPooling2D((3, 3), strides=2)(inc_4e)

inc_5a = getInception(256, 160, 320, 32, 128, 128, max_3x3_2_4)
inc_5b = getInception(384, 192, 384, 48, 128, 128, inc_5a)

avg_6 = layers.AveragePooling2D((7, 7))(inc_5b)
dropout_6 = layers.Dropout(0.4)(avg_6)
fc_6 = layers.Dense(1000, activation='relu')(dropout_6)
softmax_2 = layers.Dense(10, activation='softmax')(fc_6)

model = tf.Model(inputs=input, outputs=softmax_2)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
