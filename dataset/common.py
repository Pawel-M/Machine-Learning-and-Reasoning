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


def create_dataset(x, y, input_dictionary, output_dictionary,
                   test_size, valid_size=None, indexed_encoding=False,
                   stratify_splits=True, random_state=1337):
    stratify = np.argmax(y, axis=1) if stratify_splits else None
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(x, y, test_size=test_size,
                                                                    random_state=random_state,
                                                                    stratify=stratify)
    if valid_size is not None:
        stratify = np.argmax(y_train_valid, axis=1) if stratify_splits else None
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=valid_size,
                                                              random_state=random_state,
                                                              stratify=stratify)
        return DeductionDataset(input_dictionary, output_dictionary, indexed_encoding,
                                x_train, y_train, x_test, y_test, x_valid, y_valid)
    else:
        return DeductionDataset(input_dictionary, output_dictionary, indexed_encoding,
                                x_train_valid, x_test, y_train_valid, y_test)


def get_dataset(folder, depth, num_variables, test_size, valid_size=None, indexed_encoding=False, random_state=1337):
    data = dataset.encoding.load_sentences_and_conclusions(folder, 'encoded_sentences', depth, num_variables)
    sentences, conclusions, input_dictionary, output_dictionary = data

    if indexed_encoding:
        sentences = ohe_to_index(sentences)

    x = kr.preprocessing.sequence.pad_sequences(sentences, padding='post')

    y = np.array(conclusions)

    print(np.argmax(y, axis=1))

    ds = create_dataset(x, y, input_dictionary, output_dictionary, test_size, valid_size, indexed_encoding,
                        stratify_splits=True, random_state=random_state)
    return ds


def get_mental_models_dataset(folder, num_variables,
                              test_size, valid_size=None,
                              indexed_encoding=False, pad_mental_models=False, random_state=1337):
    data = dataset.encoding.load_sentences_and_conclusions(folder, 'encoded_mental_models', num_variables=num_variables)
    sentences, mental_models, input_dictionary, output_dictionary = data

    if indexed_encoding:
        sentences = ohe_to_index(sentences)

    x = kr.preprocessing.sequence.pad_sequences(sentences, padding='post')
    if pad_mental_models:
        y = kr.preprocessing.sequence.pad_sequences(mental_models, padding='post')
    else:
        y = mental_models

    print(y[0:5])

    ds = create_dataset(x, y, input_dictionary, output_dictionary, test_size, valid_size, indexed_encoding,
                        stratify_splits=False, random_state=random_state)
    return ds


def get_separated_sequences_mental_models_dataset(folder, base_name, num_variables, max_depth,
                                                  test_size, valid_size=None,
                                                  indexed_encoding=False, pad_mental_models=False, random_state=1337):
    data = dataset.encoding.load_sentences_and_conclusions(folder,
                                                           base_name=base_name,
                                                           num_variables=num_variables, max_depth=max_depth)

    sentences, mental_models, input_dictionary, output_dictionary = data

    max_len = None
    if indexed_encoding:
        max_len = 0
        encoded_sentences = []
        for separated_sentences in sentences:
            encoded_separated_sentences = ohe_to_index(separated_sentences)
            for sentence in encoded_separated_sentences:
                max_len = max(max_len, sentence.shape[0])

            encoded_sentences.append(encoded_separated_sentences)

        sentences = encoded_sentences

    padded_sentences = []
    for separated_sentences in sentences:
        padded_sentences.append(kr.preprocessing.sequence.pad_sequences(separated_sentences,
                                                                        padding='post', maxlen=max_len))

    x = kr.preprocessing.sequence.pad_sequences(padded_sentences, padding='post')

    if pad_mental_models:
        max_mms_len = 0
        for mms in mental_models:
            max_mms_len = max(max_mms_len, len(mms))

        if max_mms_len > 1:
            padded_mental_models = []
            for mms in mental_models:
                padded_mental_models.append(kr.preprocessing.sequence.pad_sequences(mms, padding='post',
                                                                                    maxlen=max_mms_len))

            y = kr.preprocessing.sequence.pad_sequences(padded_mental_models, padding='post')
        else:
            y = np.array(mental_models)
    else:
        y = np.array(mental_models)

    print(x[0:5])
    print(y[0:5])

    ds = create_dataset(x, y, input_dictionary, output_dictionary, test_size, valid_size, indexed_encoding,
                        stratify_splits=False, random_state=random_state)
    return ds


def get_joined_sequences_mental_models_dataset(folder, base_name, num_variables, max_depth,
                                               test_size, valid_size=None,
                                               indexed_encoding=False, pad_mental_models=False, random_state=1337):
    data = dataset.encoding.load_sentences_and_conclusions(folder,
                                                           base_name=base_name,
                                                           num_variables=num_variables, max_depth=max_depth)

    sentences, mental_models, input_dictionary, output_dictionary = data
    and_symbol = np.array(input_dictionary['and'])

    encoded_sentences = []
    for separated_sentences in sentences:
        to_join_sentences = [separated_sentences[0]]
        for sentence_index in range(1, len(separated_sentences)):
            to_join_sentences.append(and_symbol)
            to_join_sentences.append(separated_sentences[sentence_index])

        joined_sentence = np.vstack(to_join_sentences)
        encoded_sentences.append(joined_sentence)

    if indexed_encoding:
        encoded_sentences = ohe_to_index(encoded_sentences)

    x = kr.preprocessing.sequence.pad_sequences(encoded_sentences, padding='post')

    if pad_mental_models:
        max_mms_len = 0
        for mms in mental_models:
            max_mms_len = max(max_mms_len, len(mms))

        if max_mms_len > 1:
            padded_mental_models = []
            for mms in mental_models:
                padded_mental_models.append(kr.preprocessing.sequence.pad_sequences(mms, padding='post',
                                                                                    maxlen=max_mms_len))

            y = kr.preprocessing.sequence.pad_sequences(padded_mental_models, padding='post')
        else:
            y = np.array(mental_models)
    else:
        y = np.array(mental_models)

    print(x[0:5])
    print(y[0:5])

    ds = create_dataset(x, y, input_dictionary, output_dictionary, test_size, valid_size, indexed_encoding,
                        stratify_splits=False, random_state=random_state)
    return ds


if __name__ == '__main__':
    # ds = get_mental_models_dataset('./data', num_variables=5,
    #                                test_size=.1, valid_size=.1,
    #                                indexed_encoding=True, pad_mental_models=True)
    ds_separated = get_separated_sequences_mental_models_dataset('./data', 'encoded_and_trees_single_mms',
                                                                 num_variables=5, max_depth=2,
                                                                 test_size=.1, valid_size=.1,
                                                                 indexed_encoding=True, pad_mental_models=True)

    ds_joined = get_joined_sequences_mental_models_dataset('./data', 'encoded_and_trees_single_mms',
                                                           num_variables=5, max_depth=2,
                                                           test_size=.1, valid_size=.1,
                                                           indexed_encoding=True, pad_mental_models=True)

    # [[first], [second]]
