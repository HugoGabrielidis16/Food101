import tensorflow as tf
from tensorflow.keras import layers


def base_model():
    efficient_net = tf.keras.applications.EfficientNetB1(
        include_top=False, weights="imagenet"
    )
    efficient_net.trainable = False
    input = layers.Input(shape=(224, 224, 3))
    x = efficient_net(input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(101)(x)
    output = layers.Activation("softmax", dtype=tf.float32, name="output_layer")(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model


def fine_tuned_model(layers_numbers):
    return


if __name__ == "__main__":
    model = base_model()
    for layer in model.layers:
        print(
            f"""
        Name of the layer : {layer.name} , 
        Is the layer trainable ?  {layer.trainable} """
        )
