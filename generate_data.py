import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


def image_augmentation(x):
    """
    Apply some image augmentation function a tensor

    Parameters
    ----------
    x (tensor): tensor to apply the augmentation on

    """
    img_augmentation = tf.keras.Sequential(
        [
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip(),
            layers.RandomContrast(factor=0.1),
        ]
    )
    return img_augmentation(x)


def processing_image(img, label, HEIGHT=224, WIDTH=224, augmented=False):
    """
    Function to map on each elements of the dataset.
    Turns an unknown sized tensor into a tensor (224, 224, 3), cast it to float32, and apply image augmentation to it if
    precised.


    Parameters
    ----------
    img (tensor, array): the image to convert
    label(tensor, array, int) : the label of the image, remain unchanged.
    HEIGHT (int) : the height we want for the targeted image
    WIDTH (int) : the width we want for the targeted image
    augmented (bool ) :

    """
    img = tf.image.resize(img, (HEIGHT, WIDTH))
    img = tf.cast(img, tf.float32)
    if augmented:
        img = image_augmentation(img)
    return img, label


def process_ds(ds, batch_size=32, augmented=False):
    """
    Takes as tf.data.Dataset map the processing image function to it, batch it to 32

    Parameters :
    ds (tf.data.Dataset) :
    """
    ds = ds.map(
        map_func=lambda x, y: processing_image(x, y, augmented=augmented),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = (
        ds.shuffle(buffer_size=1000)
        .batch(batch_size=batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return ds


def load_data():
    """
    Download or load the food101 datasets

    Apply
    """
    (train_data, test_data), info = tfds.load(
        name="food101",
        split=["train", "validation"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir=".",
    )
    train_data = process_ds(train_data, batch_size=32, augmented=True)
    test_data = process_ds(test_data, batch_size=64)

    return train_data, test_data, info


if __name__ == "__main__":
    train_data, test_data, info = load_data()
    class_names = info.features["label"].names
    print(class_names[:10])
    sample = train_data.take(1)
    print(sample)
    for images, labels in sample:
        print(images[0].dtype)
        print(labels[0].dtype)

    # show_image(image, class_names[label.numpy()])
