import tensorflow as tf
from tensorflow.keras import layers
import sklearn

""" from tensorflow.keras import mixed_precision  # with tf 2.8

mixed_precision.set_global_policy("mixed_float16")
mixed_precision.global_policy() """


def base_EfficientNetB2_model():
    efficient_net = tf.keras.applications.EfficientNetB2(
        include_top=False, weights="imagenet"
    )
    efficient_net.trainable = False
    input = layers.Input(shape=(224, 224, 3), dtype=tf.float32)
    x = efficient_net(input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(101)(x)
    output = layers.Activation("softmax", dtype=tf.float32, name="output_layer")(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model


def finetuned_EfficientNetB2_model(layers_numbers):
    efficient_net = tf.keras.applications.EfficientNetB2(
        include_top=False, weights="imagenet"
    )
    efficient_net.trainable = True
    for layer in efficient_net.layers[:-layers_numbers]:
        layer.trainable = False

    # Freeze BatchNorm layers
    for layer in efficient_net.layers[-layers_numbers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    input = layers.Input(shape=(224, 224, 3))
    x = efficient_net(input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(101)(x)
    output = layers.Activation("softmax", dtype=tf.float32, name="output_layer")(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model


if __name__ == "__main__":
    base_model = base_EfficientNetB2_model()
    base_model.summary()
    finetuned_model = fine_EfficientNetB2_model(12)
    finetuned_model.summary()
