import logging
import os
import subprocess
import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from tqdm import tqdm

# EDEN DUPONT 204808596


train_file = "train.txt"
validation_file = "val.txt"
test_file = "test.txt"
temp_files = []
table_names = {}
feature_names = ["Checking Status", "Saving Status", "Credit History", "Housing", "Job", "Property Magnitude", "Number of dependents", "Number Credits", "Own Telephone",
                 "Foreign Workers"]
map_to_int = {}


def encode_target(data):
    df_mod = data.copy()
    changed = False
    for key in list(data.columns[:]):
        if key not in map_to_int.keys():
            changed = True
            targets = sorted(df_mod[key].unique())
            map_to_int[key] = {name: n + 1 for n, name in enumerate(targets)}
        df_mod[key].replace(map_to_int[key], inplace=True)
    if changed:
        logging.debug(map_to_int)
    return df_mod


def get_x_y(data):
    features = list(data.columns[:-1])
    target_y = list(data.columns[-1])

    x = data[features]
    y = data[target_y]
    return x, y


# build decision tree using training data, use pandas
def decision_tree_build(encoded_train_data, encoded_valid_data, epochs=10000, min_score=90):
    logging.debug("decision tree build : begin")
    x_train, y_train = get_x_y(encoded_train_data)
    x_valid, y_valid = get_x_y(encoded_valid_data)

    clf = DecisionTreeClassifier(criterion="entropy", splitter="random")
    logging.debug(f"Training model with {epochs} epochs")
    for _ in tqdm(range(epochs), desc="Training model"):
        clf.fit(x_train, y_train)
        score = clf.score(x_valid, y_valid)
        if score > min_score:
            break
    logging.debug(f"Tree ready - validation score {score}")
    return clf


def plot_tree(tree, features):
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=features, precision=5, filled=True, rounded=True,
                        special_characters=True, class_names=["Good", "Bad"], impurity=True)

    command = ["dot", "-Tpng", "dt.dot", "-o", "tree_ex4.png"]
    try:
        subprocess.call(command, shell=True)
    except:
        logging.debug('Could not run dot, ie graphviz, to produce visualization')


def print_accuracy(tree, encoded_data):
    x, y = get_x_y(encoded_data)
    score = tree.score(x, y)
    print(f"Tree score is {score}")


def clean_file(directory):
    logging.info(msg="Cleaning file " + directory)
    names = ""
    write_first_line = True
    new_file = directory + "_temp"
    temp_files.append(new_file)
    pattern = re.compile(r",| ")
    f = open(directory, "r")
    new_f = open(new_file, "w")
    lines = f.readlines()
    for line in lines:
        line1 = pattern.split(line.rstrip("\n"))
        if line1[0] in ["##"]:
            names += line1[1] + ", "
            table_names[line1[1].rstrip()] = line1[2:]
        if line1[0] in ["%%"]:
            table_names["Y"] = line1[1:]
        if line1[0] not in ["##", "%%", "//"]:
            if write_first_line:
                write_first_line = False
                new_f.write(names + "Y\n")
            new_f.write(line)

    f.close()
    new_f.close()
    return new_file


def read_file(directory):
    cfile = clean_file(directory)
    data = pd.read_csv(cfile, header=0, sep=", |,", engine="python")
    return data


# leaving directory empty uses a relative path
def read_data(directory=""):
    logging.info(msg="Reading training file")
    train = read_file(directory + train_file)
    logging.info(msg="Reading validation file")
    valid = read_file(directory + validation_file)
    logging.info(msg="Reading test file")
    test = read_file(directory + test_file)
    return train, valid, test


def clean_temp_files():
    logging.info(msg="Deleting temporary files")
    for x in temp_files:
        os.remove(x)


def main():
    logging.basicConfig(level=logging.DEBUG)
    # Read datasets
    train, valid, test = read_data()

    # Encode alphabetic values into numeric values
    encoded_train = encode_target(data=train)
    encoded_validation = encode_target(data=valid)
    encoded_test = encode_target(data=test)

    # get a trained tree
    tree = decision_tree_build(encoded_train, encoded_validation)

    logging.debug("Calculating accuracy with test data")
    print_accuracy(tree, encoded_test)

    plot_tree(tree, feature_names)

    clean_temp_files()


if __name__ == "__main__":
    main()
