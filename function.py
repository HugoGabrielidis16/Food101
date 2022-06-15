from cProfile import label
import matplotlib.pyplot as plt
import tensorflow as tf


def show_image(image, label):

    """
    Show an image and it's label
    """

    plt.imshow(image)

    plt.title(f"""Class name : {label}""")
    plt.axis(False)
    plt.show()


def preprocessing_image(image, label, img_shape=224, scale=False):

    """
    Change the image type form uint8 to float32, reshape them to (224,224,3) and scale them between 0 and 1 if necessary

    Parameters
    ----------
    image (array) : the image to reshape & convert
    label (int): no action done on it
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default False


    Returns
    -------
    image : the modified image
    label : same as the args
    """

    # image = tf.resize(image, (img_shape, img_shape,3)) # for tf > 2.8
    image = tf.image.resize(image, [img_shape, img_shape])  # for tf = 2.4
    image = tf.cast(image, tf.float32)
    if scale:
        image = image / 255  # Not required if using EfficientNetBX models

    return image, label


def prepare_dataset(dataset, batch_size=32):

    """
    Prepare the dataset, split them into 32 size batch, shuffle them ,and preprocess the image within

    Parameters
    ---------
    dataset(tf.data.dataset) :
    batch_size(int) :
    """

    dataset = (
        dataset.map(map_func=preprocessing_image)
        .shuffle(buffer_size=1000)
        .batch(batch_size=batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return dataset
