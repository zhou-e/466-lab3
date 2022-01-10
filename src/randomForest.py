import pandas as pd
import random
import InduceC45
import classify


def randomForest(dataset, target, tree_count, attrs_per_tree, data_per_tree, dataName, attributes):
    forest = []
    features = dataset.columns.to_list()
    features.remove(target)
    for x in range(tree_count):
        tree_attrs = random.sample(features, attrs_per_tree)
        data = dataset.sample(data_per_tree, replace=True)
        tree = InduceC45.c45(data[tree_attrs], data[target], dataName, attributes)
        forest.append(tree)
    return forest


def classify_RF(forest, dataX, attributes):
    predictions = pd.DataFrame()
    for tree in forest:
        prediction = classify.classification(dataX, tree, attributes)
        predictions = pd.concat([predictions, pd.Series(prediction)], axis=1)
    final_predictions = predictions.mode(axis=1)
    return final_predictions


def main():
    #filename = sys.argv[1]
    #num_attributes = sys.argv[2]
    #num_data_points = sys.argv[3]
    #num_trees = sys.argv[4]

    #filename = "part2/iris.data.csv"
    filename = "part2/letter-recognition.data.csv"
    num_attributes = 16
    num_data_points = 2000
    num_trees = 1

    df = pd.read_csv(filename)
    target = df.iloc[1][0]
    df = df.drop(1)
    if df.iloc[0, 0] == '0':
        df.iloc[:, 0] = df.iloc[:, 0].astype(float)

    attributes = {x:df[x].iloc[2:].unique() if df[x][0] != 0 else None for x in df.columns}
    df = df.iloc[1:,:]

    forest = randomForest(df, target, num_trees, num_attributes, num_data_points, filename, attributes)
    classified = classify_RF(forest, df.drop(target, axis=1), attributes)
    results = classify.get_confusion_matrix(classified[0].values, df[target], df[target])
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
