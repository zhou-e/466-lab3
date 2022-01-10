import pandas as pd
import numpy as np
import math
from collections import OrderedDict
import sys
import json


def find_best_split(data, feature, target):
    data_entropy = get_entropy(data, target)
    df = data.sort_values(feature)
    df_counts = df[target].value_counts()
    left_side = [df[df[feature] <= x] for x in df[feature].unique()]
    sides = [np.array([x[target].value_counts(), np.nan_to_num(df_counts - x[target].value_counts(), nan=df_counts)]) for x in left_side]
    entropies = [np.array([((-x[0]/x[0].sum())*np.log2(x[0]/x[0].sum())).sum(), ((-x[1]/x[1].sum())*np.ma.log2(x[1]/x[1].sum())).sum()]) for x in sides[:-1]]
    split_entropies = np.array([(sides[x][0].sum()/len(df)) * entropies[x][0] + ((len(df) - sides[x][0].sum())/len(df)) * entropies[x][1] for x in range(len(entropies))])
    if split_entropies.size == 0:
        return np.NINF, None
    alpha = left_side[split_entropies.argmin()][feature].iloc[-1]
    return data_entropy - split_entropies.min(), alpha


def get_entropy(df, target, selected_feature=None):
    if selected_feature is None:
        target_column = df[target]
        entropy = 0
        for category in target_column.unique():
            p = target_column[target_column == category].count() / len(target_column)
            entropy += -p * math.log(p, 2)
        return entropy
    tot_entropy = 0
    for category in df[selected_feature].unique():
        subset = df[df[selected_feature] == category][target]
        entropy = 0
        for clas in subset.unique():
            p = subset[subset == clas].count() / len(subset)
            entropy += -p * math.log(p, 2)
        tot_entropy += (len(subset) / len(df[target])) * entropy
    return tot_entropy


def select_split(df, features, target, attributes):
    data_entropy = get_entropy(df, target)
    max_gain = np.NINF
    best_feature = None
    for feature in features:
        if attributes[feature] is not None:
            feat_entropy = get_entropy(df, target, feature)
            feat_gain = data_entropy - feat_entropy
        else:
            feat_gain, alpha = find_best_split(df, feature, target)
        if feat_gain > max_gain:
            max_gain = feat_gain
            best_feature = feature
    return max_gain, best_feature
      

def split_df(df, value, selected_feature, le = None):
    if le:
        return df[df[selected_feature] <= value]
    elif le is False:
        return df[df[selected_feature] > value]
    return df[df[selected_feature] == value]


def build_model(df, features, target, cut_off, attributes):
    def get_setup_variables(var):
        current_tree = OrderedDict({"var": var, "edges": []})
        return current_tree

    def find_best_category(features = None, under_threshold = False, le = False, gt = False):
        best = -1
        if under_threshold:
            subset = target_column
        elif le:
            subset = target_column[new_df[features] <= value]
        elif gt:
            subset = target_column[new_df[features] > value]
        else:
            subset = target_column[new_df[features] == value]
        for category in target_categories:
            count = subset[subset == category].count() / len(subset)
            if count > best:
                best = count
                best_decision = category
        return best, best_decision

    new_df = df.copy()
    target_column = new_df[target]
    target_categories = target_column.unique()


    # if there is no difference in target
    if len(target_categories) == 1: 
        decision = target_categories[0]
        current_tree = OrderedDict({"decision":decision, "p": 1.0})
        return current_tree, True


    max_gain, selected_feature = select_split(new_df, features, target, attributes)
    # if the information gain doesn't exceed the threshold
    if max_gain < cut_off: 
        # find most likely target based on feature selections, set as leaf
        best, best_decision = find_best_category(under_threshold=True)
        current_tree = OrderedDict({"decision": best_decision, "p": best})
        return current_tree, True
        
        # if there is only one feature to split by
    if len(features) == 1:
        current_tree = get_setup_variables(features[0])
        # Categorical Variable
        if attributes[features[0]] is not None:
            for value in attributes[features[0]]:
                if len(new_df[new_df[features[0]] == value]) > 0:
                    best, best_decision = find_best_category(features[0])
                    current_tree["edges"].append(OrderedDict({"edge": OrderedDict({"value": value, "leaf": OrderedDict({"decision": best_decision, "p": best})})}))
                else:
                    best, best_decision = find_best_category(under_threshold=True)
                    current_tree["edges"].append(OrderedDict({"edge": OrderedDict({"value": value, "leaf": OrderedDict({"decision": best_decision, "p": best})})}))
        # Numeric Variable
        else:
            gain, value = find_best_split(df, features[0], target)
            best, best_decision = find_best_category(features[0], under_threshold=True, le=True)
            current_tree["edges"].append(OrderedDict({"edge": OrderedDict({"value": value, "direction": "le", "leaf": OrderedDict({"decision": best_decision, "p": best})})}))
            best, best_decision = find_best_category(features[0], under_threshold=True, gt=True)
            current_tree["edges"].append(OrderedDict({"edge": OrderedDict({"value": value, "direction": "gt", "leaf": OrderedDict({"decision": best_decision, "p": best})})}))
        return current_tree, False

    # create node with selected_feature
    current_tree = get_setup_variables(selected_feature)
    if attributes[selected_feature] is not None:
        for value in attributes[selected_feature]:
            split = split_df(new_df, value, selected_feature)
            if len(split) > 0: 
                next_edge, indic = build_model(split, [feature for feature in features if feature != selected_feature], target, cut_off, attributes)
                if indic:
                    current_tree["edges"].append(OrderedDict({"edge": OrderedDict({"value": value, "leaf": next_edge})}))
                else:
                    current_tree["edges"].append(OrderedDict({"edge": OrderedDict({"value": value, "node": next_edge})}))
            else:
                best, best_decision = find_best_category(under_threshold=True)
                current_tree["edges"].append(OrderedDict({"edge": OrderedDict({"value": value, "leaf": OrderedDict({"decision": best_decision, "p": best})})}))
    else:
        gain, value = find_best_split(df, selected_feature, target)
        for i in range(2):
            if i==0:
                direction = "le"
            else:
                direction = "gt"
            split = split_df(new_df, value, selected_feature, i==0)
            if split.equals(new_df):
                best, best_decision = find_best_category(under_threshold=True)
                return OrderedDict({"decision": best_decision, "p": best}), True
            next_edge, indic = build_model(split, features, target, cut_off, attributes)
            if indic:
                current_tree["edges"].append(OrderedDict({"edge": OrderedDict({"value": value, "direction": direction, "leaf": next_edge})}))
            else:
                current_tree["edges"].append(OrderedDict({"edge": OrderedDict({"value": value, "direction": direction, "node": next_edge})}))
    return current_tree, False


def c45(dataX, dataY, dataName, attributes):
    data = pd.concat([dataX, dataY], axis=1)
    tree, n = build_model(data, dataX.columns.to_list(), dataY.name, .3, attributes)
    if "decision" in tree:
        return OrderedDict({"dataset": dataName, "leaf": tree})
    return OrderedDict({"dataset": dataName, "node": tree})


def main():
    #data = pd.read_csv(sys.argv[1])
    #target = data.iloc[1][0]
    #data = data.iloc[2:,:]

    data = pd.read_csv("part2/iris.data.csv")
    #data = pd.read_csv("part2/iris.data.csv")
    target = data.iloc[1][0]
    data = data.drop(1)
    # Convert first column back to float
    if data.iloc[0, 0] == '0':
        data.iloc[:, 0] = data.iloc[:, 0].astype(float)

    attributes = {x:data[x].iloc[2:].unique() if data[x][0] != 0 else None for x in data.columns}
    data = data.iloc[1:,:]

    if len(sys.argv) == 3:
        columns = data.columns
        restrictions_file = sys.argv[2]
        restrictions = []
        with open(restrictions_file, "r") as file:
            i = 0
            for line in file:
                if line.strip() == "1":
                    restrictions.append(columns[i])
                i += 1
        restricted_df = data[restrictions]
    else:
        restricted_df = data
    #data_name = sys.argv[1]
    tree = c45(restricted_df.drop(target, axis=1), restricted_df[target], "DATA", attributes)
    print(json.dumps(tree, indent=2))


if __name__ == "__main__":
    main()