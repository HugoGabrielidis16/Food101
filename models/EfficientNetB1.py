import tensorflow as tf
from tensorflow.keras import layers


def base_EfficientNetB1_model():
    efficient_net = tf.keras.applications.EfficientNetB1(
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


def fine_EfficientNetB1_model(layers_numbers):
    efficient_net = tf.keras.applications.EfficientNetB1(
        include_top=False, weights="imagenet"
    )
    efficient_net.trainable = False

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
    model = base_EfficientNetB1_model()
    for layer in model.layers:
        print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)
