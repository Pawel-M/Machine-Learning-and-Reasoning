import numpy as np
import tensorflow.keras as kr
from sklearn.model_selection import train_test_split

import dataset.encoding


class DeductionDataset:
    def __init__(self, input_dictionary, output_dictionary, indexed_encoding,
                 x_train, y_train, x_test, y_test, x_valid=None, y_valid=None):
        self.input_dictionary = input_dictionary
        self.output_dictionary = output_dictionary
        self.indexed_encoding = indexed_encoding
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_valid = x_valid
        self.y_valid = y_valid


def ohe_to_index(sentences):
    return [np.where(sentence == 1)[1] + 1 for sentence in sentences]


def get_dataset(folder, depth, variables, test_size, valid_size=None, indexed_encoding=False, random_state=1337):
    data = dataset.encoding.load_sentences_and_conclusions(folder, depth, variables)
    sentences, conclusions, input_dictionary, output_dictionary = data

    if indexed_encoding:
        sentences = ohe_to_index(sentences)

    x = kr.preprocessing.sequence.pad_sequences(sentences, padding='post')

    y = np.array(conclusions)

    print(np.argmax(y, axis=1))

    x_train_valid, x_test, y_train_valid, y_test = train_test_split(x, y, test_size=test_size,
                                                                    random_state=random_state,
                                                                    stratify=np.argmax(y, axis=1))
    if valid_size is not None:
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=valid_size,
                                                              random_state=random_state,
                                                              stratify=np.argmax(y_train_valid, axis=1))
        return DeductionDataset(input_dictionary, output_dictionary, indexed_encoding,
                                x_train, y_train, x_test, y_test, x_valid, y_valid)
    else:
        return DeductionDataset(input_dictionary, output_dictionary, indexed_encoding,
                                x_train_valid, x_test, y_train_valid, y_test)
