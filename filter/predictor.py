import numpy as np
import pandas as pd
import os
import time
import pickle
import sys
import tensorflow as tf

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


class Predictor:
    def __init__(self):
        self.HAM = "ham"
        self.SPAM = "spam"
        self.SOURCES = [("HAM", self.HAM), ("SPAM", self.SPAM)]
        self.SKIP_FILES = {"cmds"}
        self.NEWLINE = "\n"
        self.list_of_email = []
        self.num_max = 4000
        self.spam_model = load_model("spam_detector_enron_model2.h5")
        self.graph = tf.get_default_graph()
        with open("tokenizer.pickle", "rb") as handle:
            self.tokenizer = pickle.load(handle)

    def progress(self, i, end_val, bar_length=50):
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

    def read_files(self, path):
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
                self.read_files(os.path.join(root, path))
            for file_name in file_names:
                if file_name not in self.SKIP_FILES:
                    file_path = os.path.join(root, file_name)
                    if os.path.isfile(file_path):
                        past_header, lines = False, []
                        f = open(file_path, encoding="latin-1")
                        for line in f:
                            if past_header:
                                lines.append(line)
                            elif line == self.NEWLINE:
                                past_header = True
                        f.close()
                        content = self.NEWLINE.join(lines)
                        yield file_path, content

    def build_data_frame(self, l, path, classification):
        rows = []

        index = []
        for i, (file_name, text) in enumerate(self.read_files(path)):
            if (i + l) % 100 == 0:
                self.progress(i + l, 58910, 50)
            self.list_of_email.append((text, file_name))
            rows.append({"text": text, "label": classification, "file": file_name})
            index.append(file_name)

        data_frame = pd.DataFrame(rows, index=index)
        return data_frame, len(rows)

    def load_data(self):
        data = pd.DataFrame({"text": [], "label": [], "file": []})
        l = 0
        for path, classification in self.SOURCES:
            data_frame, nrows = self.build_data_frame(l, path, classification)
            data = data.append(data_frame)
            l += nrows
        # data = data.reindex(np.random.permutation(data.index))
        return data

    def token_count(self, row):
        "returns token count"

        text = row["tokenized_text"]
        length = len(text.split())
        return length

    def tokenize(self, row):
        "tokenize the text using default space tokenizer"
        text = row["text"]
        lines = (line for line in text.split(self.NEWLINE))
        tokenized = ""
        for sentence in lines:
            tokenized += " ".join(tok for tok in sentence.split())
        return tokenized

    def train_tf_idf_model(self, texts):
        "train tf idf model "
        tic = time.process_time()

        tok = Tokenizer(num_words=self.num_max)
        tok.fit_on_texts(texts)
        toc = time.process_time()

        print(" -----total Computation time = " + str((toc - tic)) + " seconds")
        return tok

    def prepare_model_input(self, tfidf_model, dataframe, mode="tfidf"):

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

    def prediction(self, data):
        result = self.spam_model.predict(data)
        prediction = [round(x[0]) for x in result]
        return prediction

    def make_data(self):
        self.data = self.load_data()
        new_index = [x for x in range(len(self.data))]
        self.data.index = new_index

        self.data["tokenized_text"] = self.data.apply(self.tokenize, axis=1)
        self.data["token_count"] = self.data.apply(self.token_count, axis=1)
        self.data["lang"] = "en"

        self.sample_texts, self.sample_target = self.prepare_model_input(
            self.tokenizer, self.data, mode=""
        )
        print(self.sample_texts, self.sample_target)
        self.predicted_value = self.prediction(np.array(self.sample_texts))
        print(self.predicted_value)

    def result(self, filename):
        index = None
        for i, (text, file_name) in enumerate(self.list_of_email):
            if filename == file_name:
                index = i
                break
        result = self.predicted_value[index]
        return result

