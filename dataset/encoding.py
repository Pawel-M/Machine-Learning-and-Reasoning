import os
import pickle

import numpy as np

import dataset.conclusion_generation


def encode_sentence(sentence, input_dictionary):
    symbols = sentence.split(' ')
    return np.array([input_dictionary[symbol] for symbol in symbols])


def encode(sentence, conclusion, input_dictionary, output_dictionary):
    return encode_sentence(sentence, input_dictionary), np.array(output_dictionary[conclusion])


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
                                   folder, base_name, max_depth=None, num_variables=None):
    file_name = base_name
    if max_depth is not None:
        file_name += f'_depth-{max_depth}'

    if num_variables is not None:
        file_name += f'_vars-{num_variables}'

    file_name += '.pkl'

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, file_name), 'wb') as file:
        pickle.dump((sentences, conclusions, input_dictionary, output_dictionary), file)


def load_sentences_and_conclusions(folder, base_name, max_depth=None, num_variables=None):
    file_name = base_name
    if max_depth is not None:
        file_name += f'_depth-{max_depth}'

    if num_variables is not None:
        file_name += f'_vars-{num_variables}'

    file_name += '.pkl'
    with open(os.path.join(folder, file_name), 'rb') as file:
        sentences, conclusions, input_dictionary, output_dictionary = pickle.load(file)

    return sentences, conclusions, input_dictionary, output_dictionary


def encode_trees(folder, max_depth, num_variables, prefix, input_length, output_length):
    df = dataset.conclusion_generation.load_trees(folder, max_depth=max_depth, num_variables=num_variables)
    input_dictionary, output_dictionary = create_dictionaries(tuple(range(1, num_variables + 1)),
                                                              ('and', 'or', 'not', '(', ')'),
                                                              input_length, output_length)

    encoded_sentences, encoded_conclusions = encode_dataframe(df, prefix=prefix,
                                                              input_dictionary=input_dictionary,
                                                              output_dictionary=output_dictionary)

    save_sentences_and_conclusions(encoded_sentences, encoded_conclusions, input_dictionary, output_dictionary,
                                   folder=folder, base_name='encoded_sentences',
                                   max_depth=max_depth, num_variables=num_variables)


def create_decoding_dictionaries(input_dictionary, output_dictionary):
    decoding_input_dictionary = {}
    for symbol in input_dictionary:
        decoding_input_dictionary[input_dictionary[symbol]] = symbol

    decoding_output_dictionary = {}
    for symbol in output_dictionary:
        decoding_output_dictionary[output_dictionary[symbol]] = symbol

    return decoding_input_dictionary, decoding_output_dictionary


def decode_sentence(sentence, decoding_input_dictionary, indexed_encoding=False):
    if indexed_encoding:
        dictionary = {np.argmax(k) + 1: decoding_input_dictionary[k] for k in decoding_input_dictionary}
        symbols = []
        for s in sentence:
            if s == 0:
                break
            symbols.append(dictionary[s])
    else:
        symbols = [decoding_input_dictionary[tuple(s.tolist())] for s in sentence]

    return ' '.join(symbols)


def decode_conclusion(conclusion, decoding_output_dictionary):
    return decoding_output_dictionary[tuple(conclusion.tolist())]


def encode_mental_models_separated_sentences(folder, max_depth, num_variables, input_length, base_name, trees_base_name):
    df = dataset.conclusion_generation.load_trees(folder, max_depth=max_depth, num_variables=num_variables,
                                                  base_name=trees_base_name)

    input_dictionary, output_dictionary = create_dictionaries(tuple(range(1, num_variables + 1)),
                                                              ('and', 'or', 'not', '(', ')'),
                                                              input_length, num_variables)

    encoded_sentences = []
    encoded_mental_models = []
    for i in range(len(df)):
        sentences_combined = str(df['infix'][i])
        sentences = sentences_combined.split(' sep ')
        mental_models = str(df['conclusion'][i])

        encoded_sentences_combined = []
        for sentence in sentences:
            encoded_sentences_combined.append(encode_sentence(sentence, input_dictionary))

        sentence_mental_models = []
        for mental_model in mental_models.split(','):
            encoded_mental_model = np.zeros(num_variables)
            for j in range(num_variables):
                if mental_model[j].lower() == 't':
                    encoded_mental_model[j] = 1
                elif mental_model[j].lower() == 'f':
                    encoded_mental_model[j] = -1

            sentence_mental_models.append(encoded_mental_model)

        encoded_sentences.append(encoded_sentences_combined)
        encoded_mental_models.append(sentence_mental_models)

    save_sentences_and_conclusions(encoded_sentences, encoded_mental_models, input_dictionary, output_dictionary,
                                   folder=folder, base_name=base_name,
                                   max_depth=max_depth, num_variables=num_variables)


if __name__ == '__main__':
    encode_mental_models_separated_sentences('./data', 2, 5, 10,
                                             'encoded_and_trees_multiple_mms',
                                             'and_trees_multiple_mms')
    data = load_sentences_and_conclusions('./data', num_variables=5, max_depth=2,
                                          base_name='encoded_and_trees_multiple_mms')
    sentences, mental_models, input_dictionary, output_dictionary = data

    for i in range(10):
        print(sentences[i])
        print(mental_models[i])
