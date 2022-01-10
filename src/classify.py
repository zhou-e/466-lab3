import pandas as pd
import numpy as np
import json
import sys


def get_confusion_matrix(pred, act, total):
    categories = total.unique()
    table = {}
    for category_1 in categories:
        for category_2 in categories:
            count = len(np.where((pred == category_1) & (act == category_2))[0])
            if category_1 not in table:
                table[category_1] = [count]
            else:
                table[category_1].append(count)

    df = pd.DataFrame.from_dict(table, orient='index', columns=categories)
    accuracy = (np.diag(df).sum(), df.to_numpy().sum())
    error = (accuracy[1] - accuracy[0], accuracy[1])

    return {
        "table": df,
        "accuracy": accuracy,
        "error": error
    }


def get_predictions(row, trees, attributes):
    tree = trees.copy()
    if "decision" in tree:
        return tree["decision"]
    variable = tree["var"]
    tree = tree["edges"]
    observed = row[variable]
    # Check with edge is the right path
    for edge in range(0,len(tree)):
        # Categorical
        if attributes[variable] is not None:
            # Go down this path
            if tree[edge]["edge"]["value"] == observed:
                # Leaf
                if "leaf" in tree[edge]["edge"]:
                    return tree[edge]["edge"]["leaf"]["decision"]
                # Node
                return get_predictions(row, tree[edge]["edge"]["node"], attributes)
        else: 
            # Less Than or Equal To
            if tree[edge]["edge"]["direction"] == "le":
                # Go Down this path
                if observed <= tree[edge]["edge"]["value"]:
                    # Leaf
                    if "leaf" in tree[edge]["edge"]:
                        return tree[edge]["edge"]["leaf"]["decision"]
                    # Node
                    return get_predictions(row, tree[edge]["edge"]["node"], attributes)
            # Greater Than
            else:
                if observed > tree[edge]["edge"]["value"]:
                    # Leaf
                    if "leaf" in tree[edge]["edge"]:
                        return tree[edge]["edge"]["leaf"]["decision"]
                    # Node
                    return get_predictions(row, tree[edge]["edge"]["node"], attributes)


def classification(data, trees, attributes):
    predictions = []
    for i in range(len(data)):
        row = data.iloc[i]
        if "node" in trees:
            predictions.append(get_predictions(row, trees["node"], attributes))
        else:
            predictions.append(get_predictions(row, trees["leaf"], attributes))
    return np.array(predictions)


def main():
    filename = open(sys.argv[2],)
    tree = json.load(filename)
    filename.close()
    data = pd.read_csv(sys.argv[1])
    target = data.iloc[1][0]
    data = data.iloc[2:,:]
    data_y = data[target]
    data_x = data.drop(target, axis=1)
    predictions = classification(data_x, tree)
    results = get_confusion_matrix(predictions, data_y, data_y)
    print("Number of Records Classified")
    print(results['accuracy'][1])
    print()
    print("Number of Records Correctly Classified")
    print(results['accuracy'][0])
    print()
    print("Number of Records Incorrectly Classified")
    print(results["error"][0])
    print()
    print("Overall Accuracy")
    print(results["accuracy"][0] / results["accuracy"][1])
    print()
    print("Error Rate")
    print(results["error"][0] / results["error"][1])
    print()
    print("Confusion Matrix")
    print(results["table"])


if __name__ == "__main__":
    main()
