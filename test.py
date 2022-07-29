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

    parser.add_argument(
        "--GPU",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    model = tf.keras.models.load_model(f"models/saved/{args.model}.h5")
    _, test_ds, _ = load_data()
    config = Config()

    # Extract X and y from test tf.data.Dataset
    X_test = np.concatenate([x for x, _ in test_ds], axis=0)
    y_test = np.concatenate([y for _, y in test_ds], axis=0)

    # When not running on GPU only take a subsample of the set
    if not eval(args.GPU):
        X_test = X_test[:30]
        y_test = y_test[:30]

    # Predict the model
    y_pred = model.predict(X_test, verbose=1)
    y_pred = [np.argmax(x) for x in y_pred]
    model.evaluate(X_test, y_test)
    print(y_test)
    print(y_pred)
    # Will returns a dictionnary with the following metrics : Accuracy, Precision, Recall, F1-score
    results = calculate_results(y_test, y_pred)
    print(results)

    """ file_name = f"results/{args.model}_results.pickle"
    print(file_name)
    with open(file_name, "wb") as f:
        pickle.dump(results, file_name, protocol=pickle.HIGHEST_PROTOCOL) """
