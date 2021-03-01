import os
import pickle

import numpy as np
import pandas as pd

import generation


def encode(sentence, conclusion, dictionary):
    symbols = sentence.split(' ')
    return np.array([dictionary[symbol] for symbol in symbols]), np.array(dictionary[conclusion])


def create_dictionary(variables, operators, length):
    assert len(variables) + len(operators) <= length, 'Length is too small to include variables and operators!'
    zeros = list([0 for _ in range(length)])

    dictionary = {}
    for index, variable in enumerate(variables):
        one_hot_encoding = list(zeros)
        one_hot_encoding[index] = 1
        dictionary[str(variable)] = one_hot_encoding

    for index, operator in enumerate(operators):
        one_hot_encoding = list(zeros)
        one_hot_encoding[length - index - 1] = 1
        dictionary[str(operator)] = one_hot_encoding

    return dictionary


def encode_dataframe(data_frame, prefix, dictionary):
    sentence_column = 'prefix' if prefix else 'infix'
    encoded_sentences = []
    encoded_conclusions = []
    for i in range(len(data_frame)):
        sentence = data_frame[sentence_column][i]
        conclusion = str(data_frame['conclusion'][i])
        encoded_sentence, encoded_conclusion = encode(sentence, conclusion, dictionary)
        encoded_sentences.append(encoded_sentence)
        encoded_conclusions.append(encoded_conclusion)

    return encoded_sentences, encoded_conclusions


def save_sentences_and_conclusions(sentences, conclusions, dictionary, folder, max_depth, num_variables):
    file_name = f'encoded_sentences_depth-{max_depth}_vars-{num_variables}.pkl'

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, file_name), 'wb') as file:
        pickle.dump((sentences, conclusions, dictionary), file)


def encode_trees(folder, max_depth, num_variables, encoding_length):
    df = generation.load_trees(folder, max_depth=max_depth, num_variables=num_variables)
    encoding_dictionary = create_dictionary(tuple(range(1, num_variables + 1)),
                                            ('and', 'or', 'not', '(', ')'),
                                            encoding_length)

    encoded_sentences, encoded_conclusions = encode_dataframe(df, prefix=False, dictionary=encoding_dictionary)
    save_sentences_and_conclusions(encoded_sentences, encoded_conclusions, encoding_dictionary,
                                   folder='../data', max_depth=2, num_variables=5)


def load_sentences_and_conclusions(folder, max_depth, num_variables):
    file_name = f'encoded_sentences_depth-{max_depth}_vars-{num_variables}.pkl'
    with open(os.path.join(folder, file_name), 'rb') as file:
        sentences, conclusions, dictionary = pickle.load(file)

    return sentences, conclusions, dictionary


if __name__ == '__main__':
    encode_trees('../data', max_depth=2, num_variables=5, encoding_length=10)

    sentences, conclusions, dictionary = load_sentences_and_conclusions('../data', max_depth=2, num_variables=5)
    print(sentences)
    print(conclusions)
    print(dictionary)

