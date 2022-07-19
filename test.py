import tensorflow as tf
from generate_data import load_data
import argparse
from config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOOD101")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    model = tf.keras.models.load_model(f"models/saved/{args.model}.h5")
    _, test_data, _ = load_data()
    config = Config()
    model.evaluate(test_data, batch_size=config.test_batch_size)
