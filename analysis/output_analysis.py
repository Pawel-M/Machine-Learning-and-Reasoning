import os

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from dataset.common import get_dataset, ohe_to_index
from dataset.encoding import encode_sentence, decode_sentence


def _plot_output_partial(ax, sentence, model,
                         labels=None, input_dictionary=None, decoding_input_dictionary=None, indexed_encoding=False):
    if input_dictionary is not None:
        title = sentence
        x = encode_sentence(sentence, input_dictionary)
        if indexed_encoding:
            x = ohe_to_index([x])[0]
    elif decoding_input_dictionary is not None:
        title = decode_sentence(sentence, decoding_input_dictionary, indexed_encoding)
        x = sentence
    else:
        title = 'Model prediction'
        x = sentence

    pred = model.predict(x[np.newaxis, ...])[0]

    if labels is None:
        labels = range(1, pred.size + 1)

    ax.bar(labels, pred)
    ax.set_title(title)
    return title


def plot_output(sentence, model,
                labels=None, input_dictionary=None, decoding_input_dictionary=None, indexed_encoding=False,
                show=True, save=False, base_dir='./'):
    fig, ax = plt.subplots()
    title = _plot_output_partial(ax, sentence, model,
                                 labels, input_dictionary, decoding_input_dictionary, indexed_encoding)

    if save:
        plt.savefig(os.path.join(base_dir, f'{title}.png'))

    if show:
        plt.show()


def plot_output_sequence(sentence, model,
                         labels=None, input_dictionary=None, decoding_input_dictionary=None, indexed_encoding=False,
                         show=True, save=False, base_dir='./'):
    partial_sentences = []
    if input_dictionary is not None:
        sentence_symbols = sentence.split(' ')
        for i in range(1, len(sentence_symbols) + 1):
            partial_sentences.append(' '.join(sentence_symbols[:i]))
    else:
        for i in range(1, sentence.shape[0] + 1):
            if (indexed_encoding and sentence[i - 1] == 0) or (not indexed_encoding and np.all(sentence[i - 1] == 0)):
                break
            partial_sentences.append(sentence[:i])

    if len(partial_sentences) == 1:
        plot_output(sentence, model, labels, input_dictionary, decoding_input_dictionary, indexed_encoding,
                    show, save, base_dir)
        return

    fig_height = len(partial_sentences) * 2
    fig, axs = plt.subplots(len(partial_sentences), 1, figsize=(3.5, fig_height))
    title = 'figure'
    for i in range(len(partial_sentences)):
        title = _plot_output_partial(axs[i], partial_sentences[i], model,
                                     labels, input_dictionary, decoding_input_dictionary, indexed_encoding)

    fig.tight_layout(h_pad=1)
    top = min(0.9 + len(partial_sentences) * .01, 0.98)
    plt.subplots_adjust(top=top)

    if save:
        plt.savefig(os.path.join(base_dir, f'{title}.png'))

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    import models.rnn_example
    import dataset.encoding as enc

    dataset = get_dataset('../data', depth=2, variables=5, test_size=.1, indexed_encoding=True)
    x_train, x_test, y_train, y_test, input_dictionary, output_dictionary = dataset

    model = models.rnn_example.load_model('../results/lstm_2021-03-11 17-00-45.414892.h5')
    indexed_encoding = True

    decoding_input_dictionary, decoding_output_dictionary = enc.create_decoding_dictionaries(input_dictionary,
                                                                                             output_dictionary)

    for sentence in tqdm.tqdm(x_test):
        plot_output_sequence(sentence, model,
                             decoding_input_dictionary=decoding_input_dictionary, indexed_encoding=indexed_encoding,
                             show=False, save=True, base_dir='../results/outputs/')
        decoded_sentence = enc.decode_sentence(sentence, decoding_input_dictionary, indexed_encoding=indexed_encoding)
        if decoded_sentence[0] == '(':
            no_brackets_sentence = ' '.join(decoded_sentence.split(' ')[1:-1])
            encoded_no_brackets_sentence = ohe_to_index([enc.encode_sentence(no_brackets_sentence, input_dictionary)])[
                0]
            plot_output_sequence(encoded_no_brackets_sentence, model,
                                 decoding_input_dictionary=decoding_input_dictionary, indexed_encoding=indexed_encoding,
                                 show=False, save=True, base_dir='../results/outputs/')

    # plot_output_sequence(x_test[0], model, decoding_input_dictionary=decoding_input_dictionary, indexed_encoding=True, show=True, save=False,
    #                      base_dir='../results/outputs/')
    # plot_output_sequence('not 1', model, input_dictionary=input_dictionary, indexed_encoding=True, show=False, save=True, base_dir='../results/outputs/')
    # plot_output_sequence('( 1 or 2 ) and ( 3 or 2 )', model, input_dictionary=input_dictionary, indexed_encoding=True, show=False, save=True, base_dir='../results/outputs/')
