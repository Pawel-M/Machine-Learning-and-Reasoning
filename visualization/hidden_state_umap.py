import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as kr
import umap

import models.common
from dataset.common import get_dataset
from dataset.encoding import decode_sentence, create_decoding_dictionaries


def get_correct_hidden_states(dataset, model):
    start_index = 0
    end_index = dataset.x_train.shape[0]
    indices_string = f'Dataset training indices: {start_index} - {end_index}'
    start_index = end_index
    end_index = end_index + dataset.x_test.shape[0]
    indices_string += f', test indices: {start_index} - {end_index}'

    xs_arrays = [dataset.x_train, dataset.x_test]
    ys_arrays = [dataset.y_train, dataset.y_test]
    if dataset.x_valid is not None:
        start_index = end_index
        end_index = end_index + dataset.x_valid.shape[0]
        indices_string += f', validation indices: {start_index} - {end_index}.'
        xs_arrays.append(dataset.x_valid)
        ys_arrays.append(dataset.y_valid)

    print(indices_string)

    all_xs = np.concatenate(xs_arrays, axis=0)
    all_ys = np.concatenate(ys_arrays, axis=0)
    all_preds = model.predict(all_xs)
    correct = np.argmax(all_ys, axis=1) == np.argmax(all_preds, axis=1)
    wrong = np.argmax(all_ys, axis=1) != np.argmax(all_preds, axis=1)
    correct_xs = all_xs[correct]
    correct_ys = all_ys[correct]
    correct_preds = all_preds[correct]

    print(f'Correctly labeled: {correct_ys.shape[0]} / {all_ys.shape[0]}')
    print('Incorrectly labeled:')
    print('=== Inputs:')
    print(all_xs[wrong])
    print('=== True labels:')
    print(all_ys[wrong])
    print('=== Predictions:')
    print(all_preds[wrong])

    hidden_output = model.layers[-2].output
    stripped_model = kr.Model(inputs=model.input, outputs=hidden_output)
    hidden_preds = stripped_model.predict(correct_xs)

    print('Hidden states calculated.')
    return correct_xs, correct_ys, hidden_preds, correct_preds


def calculate_projection(n_neighbors, min_dist, hidden_preds):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    embedding = reducer.fit_transform(hidden_preds)
    print(f'UMAP (neighbors: {n_neighbors}, min_dist: {min_dist}) found. Resulting shape: {embedding.shape}.')
    return reducer, embedding


def plot_embedding(embedding, correct_ys, model_name, n_neighbors, min_dist, save_folder=None):
    ys_indexed = np.argmax(correct_ys, axis=1)

    plt.figure(figsize=(5, 4))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        marker='.',
        c=[['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple'][x] for x in ys_indexed])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'UMAP (neighbors: {n_neighbors}, min_dist: {min_dist}) of the {model_name}')
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, f'UMAP_{model_name}_neighbors-{n_neighbors}_min_dist-{min_dist}.png'))
    plt.show()


def create_umaps(model_name, n_neighbors_values, min_dist_values, save_file=None):
    umaps = []
    for n_neighbors in n_neighbors_values:
        for min_dist in min_dist_values:
            reducer, embedding = calculate_projection(n_neighbors=n_neighbors, min_dist=min_dist,
                                                      hidden_preds=hidden_preds)
            umaps.append((reducer, embedding, f'{model_name} UMAP, n_neighbors={n_neighbors}, min_dist={min_dist}'))
            plot_embedding(embedding, correct_ys, model_name=model_name, n_neighbors=n_neighbors, min_dist=min_dist,
                           save_folder='../results/umap')

    if save_file is not None:
        with open(save_file, 'wb') as f:
            pickle.dump(umaps, f)

    return umaps


def analyze_results_in_range(dataset, xs, preds, embeddings, x_range, y_range):
    dec_input, dec_output = create_decoding_dictionaries(dataset.input_dictionary, dataset.output_dictionary)
    close_indices = []
    close_predictions = []
    print(f'index, input => max prediction probability')
    for i in range(embeddings.shape[0]):
        if x_range[0] < embeddings[i, 0] < x_range[1] \
                and y_range[0] < embeddings[i, 1] < y_range[1]:
            close_indices.append(i)
            close_predictions.append(np.max(preds[i]))
            print(f'{i}, {decode_sentence(xs[i], dec_input, dataset.indexed_encoding)} => {np.max(preds[i])}')

    plt.plot(embeddings[close_indices][:, 0], embeddings[close_indices][:, 1], linestyle='', marker='.')
    plt.show()

    plt.bar(range(len(close_predictions)), close_predictions)
    plt.show()
    return close_indices, close_predictions


if __name__ == '__main__':
    dataset = get_dataset('../data', depth=2, num_variables=5, test_size=.1, valid_size=.1, indexed_encoding=True)
    model = models.common.load_model('../results/lstm_2021-04-06 12-25-35.811640.h5')

    correct_xs, correct_ys, hidden_preds, correct_preds = get_correct_hidden_states(dataset, model)

    model_name = 'LSTM'
    n_neighbors_values = [5, 10, 15, 20, 50]
    min_dist_values = [0.01, 0.1, 0.2, 0.3, 0.5]
    save_file = '../results/umap/lstm umaps.pkl'
    # umaps = create_umaps(model_name, n_neighbors_values, min_dist_values, save_file)
    with open(save_file, 'rb') as f:
        umaps = pickle.load(f)

    u = umaps[18]
    embeddings = u[1]
    close_indices, close_predictions = analyze_results_in_range(dataset, correct_xs, correct_preds, embeddings,
                                                                x_range=(4, 8), y_range=(2.5, 6.5))
