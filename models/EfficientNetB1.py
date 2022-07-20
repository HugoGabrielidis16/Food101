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


def finetuned_EfficientNetB1_model():
    efficient_net = tf.keras.applications.EfficientNetB1(
        include_top=False, weights="imagenet"
    )
    efficient_net.trainable = True
    input = layers.Input(shape=(224, 224, 3))
    x = efficient_net(input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(101)(x)
    output = layers.Activation("softmax", dtype=tf.float32, name="output_layer")(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model


if __name__ == "__main__":
    base_model = base_EfficientNetB1_model()
    base_model.summary()
    finetuned_model = finetuned_EfficientNetB1_model()
    finetuned_model.summary()
