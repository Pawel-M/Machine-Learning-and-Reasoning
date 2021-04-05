import datetime
import itertools
import os
import time

import numpy as np
import tensorflow.keras as kr
import matplotlib.pyplot as plt
from numpy import number

from dataset.common import get_dataset
import models.common


def train_model(model, dataset, learning_rate, batch_size, epochs, patience=None, min_delta=0.):
    model.compile(optimizer=kr.optimizers.Adam(learning_rate=learning_rate),
                  loss=kr.losses.categorical_crossentropy,
                  metrics=['categorical_accuracy'])

    callbacks = []
    if patience is not None:
        assert dataset.x_valid is not None or dataset.y_valid is not None, \
            "Validation subset must not be empty when using early stopping."
        callbacks.append(kr.callbacks.EarlyStopping(patience=patience,
                                                    min_delta=min_delta,
                                                    restore_best_weights=True,
                                                    verbose=1))

    start_time = time.time()
    history = model.fit(dataset.x_train, dataset.y_train, validation_data=(dataset.x_valid, dataset.y_valid),
                        batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    end_time = time.time()

    print(f'Training time: {(end_time - start_time):.1f}s')
    pred = model.predict(dataset.x_test)
    test_accuracy = np.mean(kr.metrics.categorical_accuracy(dataset.y_test, pred))
    print(f'Test accuracy: {test_accuracy * 100:.1f} %')
    return history, test_accuracy


def plot_loss_accuracy(history):
    loss = history.history['loss']
    accuracy = history.history['categorical_accuracy']

    fig, ax = plt.subplots()
    ax.plot(range(len(loss)), loss, label='loss', color='tab:blue')
    ax.set_ylabel('loss', color='tab:blue')

    ax2 = ax.twinx()
    ax2.plot(range(len(accuracy)), accuracy, color='tab:orange')
    ax2.set_ylabel('accuracy', color='tab:orange')

    ax.set_xlabel('epoch')
    plt.show()


def train_multiple_runs(num_runs, dataset, model_fn, model_args, training_args,
                        base_name, plot_history=True, save_folder=None):
    results_folder = None
    if save_folder is not None:
        now = datetime.datetime.now()
        timestamp = str(now).replace(':', '-')
        results_folder = os.path.join(save_folder, f'results_{timestamp}')
        os.makedirs(results_folder)

    if 'input_dim' not in model_args:
        if dataset.indexed_encoding:
            model_args['input_dim'] = np.max(dataset.x_train) + 1
        else:
            model_args['input_dim'] = dataset.x_train.shape[-1]
            
    if 'output_dim' not in model_args:
        model_args['output_dim'] = dataset.y_train.shape[-1]



    histories = []
    accuracies = []
    result_str = ''
    for run in range(num_runs):
        print(f'Training run {run + 1}/{num_runs}')
        model = model_fn(**model_args)
        history, test_accuracy = train_model(model, dataset, **training_args)
        histories.append(history)
        accuracies.append(test_accuracy)
        print(f'Test accuracy (run {run + 1}): {test_accuracy:.2}')
        result_str += f'{test_accuracy}\t'

        if plot_history:
            plot_loss_accuracy(history)

        if results_folder is not None:
            models.common.save_model(model, results_folder, base_name)

    lines = []
    lines.append(f'Training results of the model "{base_name}"\n\n')
    lines.append(f'num_runs = {num_runs}\n')
    for key, value in itertools.chain(model_args.items(), training_args.items()):
        lines.append(f'{key} = {value}\n')

    lines.append('\n####################################\n')
    lines.append('Test accuracies:\n')
    lines.append(result_str + '\n')

    print('\n\n####################################\n')
    for line in lines:
        print(line[:-1])

    if results_folder is not None:
        with open(os.path.join(results_folder, 'results.txt'), 'w') as f:
            f.writelines(lines)

    return histories, accuracies


if __name__ == '__main__':
    dataset = get_dataset('../data', depth=2, variables=5, test_size=.1, valid_size=.1, indexed_encoding=True)

    import models.rnn_example

    model_fn = models.rnn_example.create_lstm_model
    base_name = 'lstm'
    model_args = {
        'num_layers': 1,
        'embedding_size': 64,
        'hidden_units': 128,
        'bidirectional': False,
    }
    training_args = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 2,
        'patience': 10,
        'min_delta': 1e-4,
    }
    num_runs = 3
    train_multiple_runs(num_runs, dataset, model_fn, model_args, training_args, base_name,
                        plot_history=True, save_folder='../results')

    # model = rnn_example.load_model('../results/lstm_2021-03-11 17-00-45.414892.h5')
