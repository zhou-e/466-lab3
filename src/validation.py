import pandas as pd
import random
import InduceC45
import classify
import knn
import sys
from randomForest import randomForest, classify_RF

# input variables
filename = sys.argv[1]
folds = int(sys.argv[2])
implementation = sys.argv[3]

neighbors = 5

num_trees = 50
num_attributes = 10
num_data_points = 400

# setting up dataframes
df = pd.read_csv(filename)
target = df.iloc[1][0]
df = df.drop(1)
df = df.dropna()
# Convert first column back to float
if df.iloc[0, 0] == '0':
    df.iloc[:, 0] = df.iloc[:, 0].astype(float)
if len(sys.argv) == 5:
    columns = df.columns
    restrictions_file = sys.argv[4]
    if not (len(restrictions_file) < 3 or "." not in restrictions_file[-4:]):
        restrictions = []
        with open(restrictions_file, "r") as file:
            i = 0
            for line in file:
                if line.strip() == "1":
                    restrictions.append(columns[i])
                i += 1
        restricted_df = df[restrictions]
elif len(sys.argv) == 6:
    if implementation == "knn":
        neighbors = int(sys.argv[5])
    else:
        num_trees = int(sys.argv[5])
    restricted_df = df
else:
    restricted_df = df

attributes = {x: restricted_df[x].iloc[2:].unique() if restricted_df[x][0] != 0 else None for x in restricted_df.columns}

non_continuous = []
continuous = []
ordinal = []
restricted_df_X = restricted_df.drop(target, axis=1)
for i, x in enumerate(restricted_df_X.columns):
    if restricted_df_X[x][0] == 0:
        if restricted_df_X.shape[0]*.999 < (restricted_df_X[x] == restricted_df_X[x].astype(int)).sum() < restricted_df_X.shape[0]*1.001:
            ordinal.append(i)
        else:
            continuous.append(i)
    else:
        non_continuous.append(i)

restricted_df = restricted_df.iloc[1:, :]
restricted_df_X = restricted_df.drop(target, axis=1)
restricted_df_y = restricted_df[target]

# consistent results
random.seed(33145)

# get folds
n = restricted_df.shape[0]
if folds == 0 or folds == 1:
    if implementation == "c45":
        model = InduceC45.c45(restricted_df_X, restricted_df_y, filename, attributes)

        predictions = classify.classification(restricted_df_X, model, attributes)
    elif implementation == "knn":
        predictions = knn.classify(restricted_df, neighbors, restricted_df_X, target, ordinal, continuous, non_continuous)
    else:
        forest = randomForest(restricted_df, target, num_trees, num_attributes, num_data_points, filename, attributes)
        classified = classify_RF(forest, restricted_df_X, attributes)
        predictions = classified[0].values
    confusion = classify.get_confusion_matrix(predictions, restricted_df_y.values, restricted_df_y)
    results = [confusion]
else:
    if folds == -1:
        divisions = [1 for _ in range(n)]
    else:
        divisions = [n // folds + (1 if x < n % folds else 0) for x in range(folds)]
    indices = [i for i in range(n)]
    random.shuffle(indices)
    results = []
    i = 0
    for div in divisions:
        test = indices[i:i+div]
        test_X = restricted_df_X.iloc[test]
        test_y = restricted_df_y.iloc[test]
        if implementation == "knn":
            predictions = knn.classify(restricted_df, neighbors, test_X, target, ordinal, continuous, non_continuous)
        else:
            train = indices[:i] + indices[i+div:]

            train_X = restricted_df_X.iloc[train]
            train_y = restricted_df_y.iloc[train]

            if implementation == "c45":
                model = InduceC45.c45(train_X, train_y, filename, attributes)
                predictions = classify.classification(test_X, model, attributes)
            else:
                forest = randomForest(restricted_df.iloc[train], target, num_trees, num_attributes, num_data_points, filename, attributes)
                classified = classify_RF(forest, test_X, attributes)
                predictions = classified[0].values
        confusion = classify.get_confusion_matrix(predictions, test_y.values, restricted_df_y)
        results.append(confusion)
        i += div

table = results[0]["table"]
accuracy = results[0]["accuracy"]
error = results[0]["error"]
avg_acc = results[0]["accuracy"][0] / max(1, results[0]["accuracy"][1])
avg_error = 1 - avg_acc
i = 1

acc_over_folds = [avg_acc]
for res in results[1:]:
    i += 1
    table += res["table"]
    accuracy = (accuracy[0] + res["accuracy"][0], accuracy[1] + res["accuracy"][1])
    error = (error[0] + res["error"][0], error[1] + res["error"][1])
    avg_acc += res["accuracy"][0] / max(1, res["accuracy"][1])
    avg_error += res["error"][0] / max(1, res["error"][1])
    acc_over_folds.append(res["accuracy"][0] / max(1, res["accuracy"][1]))
avg_acc /= i
avg_error /= i
print("Confusion Matrix:")
print(table)
print()
print("Overall Accuracy:")
print(accuracy[0] / accuracy[1])
print()
print("Overall Error:")
print(error[0] / error[1])
print()
print("Average Accuracy:")
print(avg_acc)
print()
print("Average Error:")
print(avg_error)

import matplotlib.pyplot as plt
fig7, ax7 = plt.subplots()
ax7.set_title('Multiple Samples with Different sizes')
ax7.boxplot([acc_over_folds] * 3)
plt.ylabel("Accuracy")
plt.show()
