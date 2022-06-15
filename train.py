from calendar import EPOCH
from distutils.command.build import build
from gc import callbacks
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from model import base_model, fine_tuned_model
from generate_data import train_data, test_data
from function import *


# Parameters
EPOCHS = 10
CHECKPOINT_PATH = "model_checkpoint/cp.cpkt"

# wandb initialisation
wandb.init(project="my-test-project", entity="yuuuugo")
wandb.config["learnig_rate"] = 0.001
wandb.config["epochs"] = 10
wandb.config["batch_size"] = 32


if __name__ == "__main__":
    train_ds = prepare_dataset(train_data)
    test_ds = prepare_dataset(test_data)

    model = base_model()

    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="vall_acc",
            save_best_only=True,
            save_weights_only=True,
        ),
        WandbCallback(),
    ]
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=wandb.config["epochs"],
        steps_per_epoch=len(train_ds),
        validation_steps=int(0.15 * len(test_ds)),
        callbacks=callbacks,
        batch_size=wandb.config["batch_size"],
    )
    model.save("Efficient_Net1-acc70.h5")
