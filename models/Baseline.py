import tensorflow as tf
from tensorflow.keras import layers


def baseline():
    inputs = layers.Input((224, 224, 3))

    x = layers.Conv2D(64, 2, activation="relu")(inputs)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(128, 2, activation="relu")(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(101)(x)
    outputs = layers.Activation("softmax", dtype=tf.float32, name="output_layer")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
