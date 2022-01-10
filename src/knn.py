import numpy as np


def get_distance(dataset, target_row, ord_columns, cont_columns, non_numeric_columns):
    ord_results = cont_results = results = 0
    target_row_new = np.array(target_row)
    # Manhattan
    if len(ord_columns) > 0:
        ord_results = (dataset.iloc[:, ord_columns] - target_row_new[ord_columns].astype(float)).abs()

    # Euclidian
    if len(cont_columns) > 0:
        cont_results = (dataset.iloc[:, cont_columns] - target_row_new[cont_columns].astype(float)) ** 2

    # Dice
    if len(non_numeric_columns) > 0:
        results = (dataset.iloc[:, non_numeric_columns] != target_row_new[non_numeric_columns]).astype(int)

    if not isinstance(ord_results, int):
        ord_results = np.sqrt(ord_results.sum(axis=1))
    if not isinstance(cont_results, int):
        cont_results = np.sqrt(cont_results.sum(axis=1))
    if not isinstance(results, int):
        results /= len(non_numeric_columns)

    return ord_results + cont_results + results


def knn(dataset, no_target, k, target_row, target_column, ord_columns, cont_columns, non_numeric_columns):
    distances = get_distance(no_target, target_row, ord_columns, cont_columns, non_numeric_columns)
    distances = distances.reset_index()
    distances = distances.sort_values(distances.columns[-1])

    # drop the first because it's the same observation
    pred = dataset.loc[distances["index"].values[1:k], :][target_column].value_counts().idxmax()
    return pred


def classify(dataset, k, test_X, target_column, ord_columns, cont_columns, non_numeric_columns):
    preds = []
    no_target = dataset.drop(target_column, axis=1)
    for row in test_X.itertuples():
        preds.append(knn(dataset, no_target, k, row[1:], target_column, ord_columns, cont_columns, non_numeric_columns))

    return np.array(preds)
