import logging
import os
import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

train_file = "train.txt"
validation_file = "val.txt"
test_file = "test.txt"
temp_files = []
table_names = {}


def encode_target(data):
    df_mod = data.copy()
    for key in table_names.keys():
        targets = sorted(df_mod[key].unique())
        map_to_int = {name: n + 1 for n, name in enumerate(targets)}
        df_mod[key].replace(map_to_int, inplace=True)
    return df_mod


# build decision tree using training data, use pandas
def decision_tree_build(data):
    encoded_data = encode_target(data=data)
    features = list(encoded_data.columns[:-1])
    target_y = list(encoded_data.columns[-1])
    x = encoded_data[features]
    y = encoded_data[target_y]
    clf = DecisionTreeClassifier(criterion="entropy", splitter="best")
    clf.fit(x, y)
    return clf


def plot_tree():
    pass


def print_accuracy():
    pass


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
    data = pd.read_csv(cfile, header=0, sep=", |,")
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
    train, valid, test = read_data()
    for name, values in table_names.items():
        text = str(name) + ": " + str(values)
        logging.info(text)
    logging.info("\n" + str(train.head(n=5)))
    tree = decision_tree_build(train)
    clean_temp_files()


if __name__ == "__main__":
    main()
