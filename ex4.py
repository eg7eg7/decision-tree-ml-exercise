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
    map_to_int = {}
    for key in table_names.keys():
        targets = table_names[key]
        map_to_int[key] = {name: (n+1) for n, name in enumerate(targets)}
        df_mod.replace(map_to_int[key], inplace=True)
    return df_mod


# build decision tree using training data, use pandas
def decision_tree_build(data):
    encoded_data = encode_target(data=data)

    print(encoded_data.head(n=6))
    clf = DecisionTreeClassifier(criterion="entropy", splitter="best")
    return 1


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


def main():
    logging.basicConfig(level=logging.DEBUG)
    train, valid, test = read_data()
    for name, values in table_names.items():
        text = str(name) + ": " + str(values)
        logging.info(text)
    logging.info("\n" + str(train.head(n=5)))
    tree = decision_tree_build(train)
    clean_temp_files()


def clean_temp_files():
    logging.info(msg="Deleting temporary files")
    for x in temp_files:
        os.remove(x)


if __name__ == "__main__":
    main()
