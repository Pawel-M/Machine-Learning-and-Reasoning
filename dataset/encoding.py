import os
import pickle

import numpy as np
import pandas as pd

import generation




def encode(sentence, conclusion, input_dictionary, output_dictionary):
    symbols = sentence.split(' ')
    return np.array([input_dictionary[symbol] for symbol in symbols]), np.array(output_dictionary[conclusion])


def create_dictionaries(variables, operators, input_length, output_length):
    assert len(variables) + len(operators) <= input_length, \
        'Input length is too small to include variables and operators!'
    assert len(variables) <= output_length, \
        'Output length is too small to include variables!'

    input_zeros = list([0 for _ in range(input_length)])
    output_zeros = list([0 for _ in range(output_length)])

    input_dictionary = {}
    output_dictionary = {}
    for index, variable in enumerate(variables):
        input_encoding = input_zeros.copy()
        input_encoding[index] = 1
        output_encoding = output_zeros.copy()
        output_encoding[index] = 1

        input_dictionary[str(variable)] = tuple(input_encoding)
        output_dictionary[str(variable)] = tuple(output_encoding)

    for index, operator in enumerate(operators):
        input_encoding = input_zeros.copy()
        input_encoding[input_length - index - 1] = 1
        input_dictionary[str(operator)] = tuple(input_encoding)

    return input_dictionary, output_dictionary


def encode_dataframe(data_frame, prefix, input_dictionary, output_dictionary):
    sentence_column = 'prefix' if prefix else 'infix'
    encoded_sentences = []
    encoded_conclusions = []
    for i in range(len(data_frame)):
        sentence = data_frame[sentence_column][i]
        conclusion = str(data_frame['conclusion'][i])
        encoded_sentence, encoded_conclusion = encode(sentence, conclusion, input_dictionary, output_dictionary)
        encoded_sentences.append(encoded_sentence)
        encoded_conclusions.append(encoded_conclusion)

    return encoded_sentences, encoded_conclusions


def save_sentences_and_conclusions(sentences, conclusions, input_dictionary, output_dictionary,
                                   folder, max_depth, num_variables):

    file_name = f'encoded_sentences_depth-{max_depth}_vars-{num_variables}.pkl'

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, file_name), 'wb') as file:
        pickle.dump((sentences, conclusions, input_dictionary, output_dictionary), file)


def encode_trees(folder, max_depth, num_variables, prefix, input_length, output_length):
    df = generation.load_trees(folder, max_depth=max_depth, num_variables=num_variables)
    input_dictionary, output_dictionary = create_dictionaries(tuple(range(1, num_variables + 1)),
                                                              ('and', 'or', 'not', '(', ')'),
                                                              input_length, output_length)

    encoded_sentences, encoded_conclusions = encode_dataframe(df, prefix=prefix,
                                                              input_dictionary=input_dictionary,
                                                              output_dictionary=output_dictionary)

    save_sentences_and_conclusions(encoded_sentences, encoded_conclusions, input_dictionary, output_dictionary,
                                   folder='../data', max_depth=2, num_variables=5)


def load_sentences_and_conclusions(folder, max_depth, num_variables):
    file_name = f'encoded_sentences_depth-{max_depth}_vars-{num_variables}.pkl'
    with open(os.path.join(folder, file_name), 'rb') as file:
        sentences, conclusions, input_dictionary, output_dictionary = pickle.load(file)

    return sentences, conclusions, input_dictionary, output_dictionary


def create_decoding_dictionaries(input_dictionary, output_dictionary):
    decoding_input_dictionary = {}
    for symbol in input_dictionary:
        decoding_input_dictionary[input_dictionary[symbol]] = symbol

    decoding_output_dictionary = {}
    for symbol in output_dictionary:
        decoding_output_dictionary[output_dictionary[symbol]] = symbol

    return decoding_input_dictionary, decoding_output_dictionary


def decode_sentence(sentence, decoding_input_dictionary):
    symbols = [decoding_input_dictionary[tuple(s.tolist())] for s in sentence]
    return ' '.join(symbols)


def decode_conclusion(conclusion, decoding_output_dictionary):
    return decoding_output_dictionary[tuple(conclusion.tolist())]


if __name__ == '__main__':
    encode_trees('../data', max_depth=2, num_variables=5, prefix=False, input_length=10, output_length=5)

    sentences, conclusions, input_dictionary, output_dictionary = load_sentences_and_conclusions('../data',
                                                                                                 max_depth=2,
                                                                                                 num_variables=5)
    print(sentences)
    print(conclusions)
    print(input_dictionary)
    print(output_dictionary)

    decoding_input_dictionary, decoding_output_dictionary = create_decoding_dictionaries(input_dictionary,
                                                                                         output_dictionary)

    decoded_sentence = decode_sentence(sentences[1], decoding_input_dictionary)
    decoded_conclusion = decode_conclusion(conclusions[1], decoding_output_dictionary)
    print(decoded_sentence)
    print(decoded_conclusion)
