import tensorflow as tf
from models.Baseline import baseline
from models.EfficientNetB0 import base_EfficientNetB0_model
from models.VIT import VIT
from function import *
from generate_data import load_data
from config import dict
import argparse


# wandb initialisation
""" import wandb
from wandb.keras import WandbCallback

wandb.init(project="my-test-project", entity="yuuuugo")
wandb.config["learnig_rate"] = 0.001
wandb.config["epochs"] = 10
wandb.config["batch_size"] = 32
# Parameters
EPOCHS = 10
CHECKPOINT_PATH = "model_checkpoint/cp.cpkt"
 """
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOOD101")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    train_ds, test_ds, info = load_data()

    model = eval(args.model)()

    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=dict["learning_rate"]),
    )

    """ callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_acc",
            save_best_only=True,
            save_weights_only=True,
        ),
    ] """
    callbacks = []
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=dict["epochs"],
        steps_per_epoch=len(train_ds),
        validation_steps=int(0.15 * len(test_ds)),
        callbacks=callbacks,
        batch_size=dict["batch_size"],
    )
    model.save(f"{model_name}.h5")
