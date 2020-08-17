#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all required libraries

import numpy as np
import pandas as pd
import os
import time
import pickle
import sys

# sys.setrecursionlimit(1500)
#%matplotlib inline

import keras

from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import LabelEncoder


# In[2]:


# pre-process the data before sending it to the model


def progress(i, end_val, bar_length=50):
    """
    Print a progress bar of the form: Percent: [#####      ]
    i is the current progress value expected in a range [0..end_val]
    bar_length is the width of the progress bar on the screen.
    """
    percent = float(i) / end_val
    hashes = "#" * int(round(percent * bar_length))
    spaces = " " * (bar_length - len(hashes))
    sys.stdout.write(
        "\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100)))
    )
    sys.stdout.flush()


# NEWLINE = "\n"


HAM = "ham"
SPAM = "spam"

SOURCES = [("HAM", HAM), ("SPAM", SPAM)]

SKIP_FILES = {"cmds"}
NEWLINE = "\n"


# Define Fuctions to load the data into the reuired format into a dataframe


def read_files(path):
    """
    Generator of pairs (filename, filecontent)
    for all files below path whose name is not in SKIP_FILES.
    The content of the file is of the form:
        header....
        <emptyline>
        body...
    This skips the headers and returns body only.
    """
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    f = open(file_path, encoding="latin-1")
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content


list_of_email = []


def build_data_frame(l, path, classification):
    global list_of_email
    rows = []
    index = []
    for i, (file_name, text) in enumerate(read_files(path)):
        if (i + l) % 100 == 0:
            progress(i + l, 58910, 50)
        list_of_email.append(text)
        rows.append({"text": text, "label": classification, "file": file_name})
        index.append(file_name)

    data_frame = pd.DataFrame(rows, index=index)
    return data_frame, len(rows)


def load_data():
    data = pd.DataFrame({"text": [], "label": [], "file": []})
    l = 0
    for path, classification in SOURCES:
        data_frame, nrows = build_data_frame(l, path, classification)
        data = data.append(data_frame)
        l += nrows
    # data = data.reindex(np.random.permutation(data.index))
    return data


data = load_data()


# We change the dataframe index from filenames to indices here.
new_index = [x for x in range(len(data))]
data.index = new_index


# Let's define functions to add two more columns here


def token_count(row):
    "returns token count"
    text = row["tokenized_text"]
    length = len(text.split())
    return length


def tokenize(row):
    "tokenize the text using default space tokenizer"
    text = row["text"]
    lines = (line for line in text.split(NEWLINE))
    tokenized = ""
    for sentence in lines:
        tokenized += " ".join(tok for tok in sentence.split())
    return tokenized


# Add these two columns to the dataframe

data["tokenized_text"] = data.apply(tokenize, axis=1)
data["token_count"] = data.apply(token_count, axis=1)

# These is not required but add another column for language just for reference
data["lang"] = "en"


# Build a tfidf model for data
# Define the max number of features
num_max = 4000


def train_tf_idf_model(texts):
    "train tf idf model "
    tic = time.process_time()

    tok = Tokenizer(num_words=num_max)
    tok.fit_on_texts(texts)
    toc = time.process_time()

    print(" -----total Computation time = " + str((toc - tic)) + " seconds")
    return tok


def prepare_model_input(tfidf_model, dataframe, mode="tfidf"):

    "function to prepare data input features using tfidf model"
    tic = time.process_time()

    le = LabelEncoder()
    sample_texts = list(dataframe["tokenized_text"])
    sample_texts = [" ".join(x.split()) for x in sample_texts]

    targets = list(dataframe["label"])
    targets = [1.0 if x == "spam" else 0.0 for x in targets]
    sample_target = le.fit_transform(targets)

    if mode == "tfidf":
        sample_texts = tfidf_model.texts_to_matrix(sample_texts, mode="tfidf")
    else:
        sample_texts = tfidf_model.texts_to_matrix(sample_texts)

    toc = time.process_time()

    print("shape of labels: ", sample_target.shape)
    print("shape of data: ", sample_texts.shape)

    print(
        " -----total Computation time for preparing model data = "
        + str((toc - tic))
        + " seconds"
    )

    return sample_texts, sample_target


with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)


spam_model = load_model("spam_detector_enron_model2.h5")


def prediction(data):
    result = spam_model.predict(data)
    prediction = [round(x[0]) for x in result]
    return prediction


sample_texts, sample_target = prepare_model_input(tokenizer, data, mode="")

# new_types = list(sample_target)

# from . import models

# for i in range(len(list_of_email)):
#     email = list_of_email[i]
#     ttype = new_types[i]

#     np_bytes = pickle.dumps(np.array(sample_texts)[i])
#     np_base64 = base64.b64encode(np_bytes)

#     newEmail = models.Email(ttype=ttype, mail=email, converted_text=np_base64)
#     newEmail.save()

# predicted = prediction(np.array(sample_texts))


# predicted_value = spam_model.predict(np.array(sample_texts))
