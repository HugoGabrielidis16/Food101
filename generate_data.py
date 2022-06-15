import tensorflow_datasets as tfds
from function import show_image

(train_data, test_data), info = tfds.load(
    name="food101",
    split=["train", "validation"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    data_dir=".",
)

if __name__ == "__main__":
    class_names = info.features["label"].names
    print(class_names[:10])
    sample = train_data.take(1)
    for image, label in sample:
        print(
            f"""
            Image shape : {image.shape}
            Image datatype : {image.dtype}
            Class format : {label}
            """
        )
        show_image(image, class_names[label.numpy()])
