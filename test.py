import tensorflow as tf
from generate_data import load_data
import argparse
from config import Config
import numpy as np
from helper_functions import calculate_results
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOOD101")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    model = tf.keras.models.load_model(f"models/saved/{args.model}.h5")
    _, test_ds, _ = load_data()
    config = Config()
    X_test = np.concatenate([x for x, _ in test_ds], axis=0)
    y_test = np.concatenate([y for _, y in test_ds], axis=0)

    y_pred = model.predict(X_test, verbose=1)
    results = calculate_results(y_test, y_pred)

    file_name = f"results/{args.model}_results"
    with open(file_name, "wb") as f:
        pickle.dump(results, file_name)
