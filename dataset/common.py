import numpy as np
import tensorflow.keras as kr
from sklearn.model_selection import train_test_split

import dataset.encoding


def ohe_to_index(sentences):
    return [np.where(sentence == 1)[1] + 1 for sentence in sentences]


def get_dataset(folder, depth, variables, test_size, indexed_encoding=False, random_state=1337):
    data = dataset.encoding.load_sentences_and_conclusions(folder, depth, variables)
    sentences, conclusions, input_dictionary, output_dictionary = data

    if indexed_encoding:
        sentences = ohe_to_index(sentences)

    x = kr.preprocessing.sequence.pad_sequences(sentences, padding='post')

    y = np.array(conclusions)

    print(np.argmax(y, axis=1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state,
                                                        stratify=np.argmax(y, axis=1))
    return x_train, x_test, y_train, y_test, input_dictionary, output_dictionary