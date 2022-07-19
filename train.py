import tensorflow as tf
from models.Baseline import baseline
from models.EfficientNetB0 import (
    base_EfficientNetB0_model,
    finetuned_EfficientNetB0_model,
)
from models.EfficientNetB1 import (
    base_EfficientNetB1_model,
    finetuned_EfficientNetB1_model,
)
from models.EfficientNetB2 import (
    base_EfficientNetB2_model,
    finetuned_EfficientNetB2_model,
)
from models.EfficientNetB3 import (
    base_EfficientNetB3_model,
    finetuned_EfficientNetB3_model,
)
from models.EfficientNetB4 import (
    base_EfficientNetB4_model,
    finetuned_EfficientNetB4_model,
)
from models.VIT import VIT
from function import *
from generate_data import load_data
import argparse
from config import Config

from tensorflow.keras import (
    mixed_precision,
)  # with tf 2.8 -> seems to be an issue when using tf 2.9

GPUS = ["GPU:0", "GPU:1"]
mixed_precision.set_global_policy("mixed_float16")
mixed_precision.global_policy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOOD101")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    # strategy = tf.distribute.MirroredStrategy(GPUS) # No need to specify GPUs since by default the strategy will use all GPUs available
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    train_ds, test_ds, info = load_data()

    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.

        model = eval(args.model)()
        config = Config(args.model)

        model.compile(
            loss="sparse_categorical_crossentropy",
            metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        )

        model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=config.epochs,
            batch_size=config.train_batch_size,
            steps_per_epoch=len(train_ds),
            validation_steps=int(0.15 * len(test_ds)),
            callbacks=config.callbacks,
        )
        model.save(config.saving_path)
