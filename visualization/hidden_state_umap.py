import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as kr
import umap

import models.common
from dataset.common import get_dataset


def get_correct_hidden_states(dataset, model):
    xs_arrays = [dataset.x_train, dataset.x_test]
    ys_arrays = [dataset.y_train, dataset.y_test]
    if dataset.x_valid is not None:
        xs_arrays.append(dataset.x_valid)
        ys_arrays.append(dataset.y_valid)

    all_xs = np.concatenate(xs_arrays, axis=0)
    all_ys = np.concatenate(ys_arrays, axis=0)
    all_preds = model.predict(all_xs)
    correct = np.argmax(all_ys, axis=1) == np.argmax(all_preds, axis=1)
    correct_xs = all_xs[correct]
    correct_ys = all_ys[correct]
    correct_preds = all_preds[correct]

    print(f'Correctly labeled: {correct_ys.shape[0]} / {all_ys.shape[0]}')

    hidden_output = model.layers[-2].output
    stripped_model = kr.Model(inputs=model.inputs, outputs=hidden_output)
    hidden_preds = stripped_model.predict(correct_xs)

    print('Hidden states calculated.')
    return correct_xs, correct_ys, hidden_preds


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


if __name__ == '__main__':
    dataset = get_dataset('../data', depth=2, variables=5, test_size=.1, valid_size=.1, indexed_encoding=True)
    model = models.common.load_model('../results/lstm_2021-04-06 12-25-35.811640.h5')

    correct_xs, correct_ys, hidden_preds = get_correct_hidden_states(dataset, model)

    model_name = 'LSTM'
    n_neighbors_values = [5, 10, 15, 20, 50]
    min_dist_values = [0.01, 0.1, 0.2, 0.3, 0.5]

    for n_neighbors in n_neighbors_values:
        for min_dist in min_dist_values:
            reducer, embedding = calculate_projection(n_neighbors=n_neighbors, min_dist=min_dist,
                                                      hidden_preds=hidden_preds)
            plot_embedding(embedding, correct_ys, model_name='LSTM', n_neighbors=n_neighbors, min_dist=min_dist,
                           save_folder='../results/umap')
