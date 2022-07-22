import wandb
from wandb.keras import WandbCallback
import tensorflow as tf


class Config:
    learning_rate = 3e-4
    resize = (224, 224)
    pretrained = True
    epochs = 20
    number_of_GPU = len(tf.config.experimental.list_physical_devices("GPU"))
    train_batch_size = 32 * number_of_GPU  # 32 per gpu used
    test_batch_size = 64
    seed = True

    # Look into it
    n_split = 5
    split = 0.9  # Need to see the len of the two sets
    # scaler = GradScaler()
    max_gnorm = 1000

    wandb_bool = False

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="model.{epoch:02d}-{val_loss:.2f}.h5"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=3, min_lr=0.001
        ),
    ]

    def __init__(self, model_name=""):
        self.model_name = model_name
        self.saving_path = f"models/saved/{model_name}.h5"

    if wandb_bool:
        wandb.init(project="FOOD101", entity="yuuuugo")
        wandb.config["learnig_rate"] = learning_rate
        wandb.config["epochs"] = epochs
        wandb.config["batch_size"] = train_batch_size
        # Parameters
        CHECKPOINT_PATH = "model_checkpoint/cp.cpkt"
