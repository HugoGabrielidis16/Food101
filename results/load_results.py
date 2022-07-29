import pickle


if __name__ == "__main__":

    f = open("baseline_results", "rb")
    results = pickle.load(f)
    print(results)
