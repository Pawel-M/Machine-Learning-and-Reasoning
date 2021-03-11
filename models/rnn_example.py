import time
import datetime
import os

import numpy as np
import tensorflow.keras as kr
import matplotlib.pyplot as plt


from dataset.common import get_dataset


def train_model(model, dataset, learning_rate, batch_size, epochs):
    x_train, x_test, y_train, y_test, _, _ = dataset


    model.compile(optimizer=kr.optimizers.Adam(learning_rate=learning_rate),
                  loss=kr.losses.categorical_crossentropy,
                  metrics=['categorical_accuracy'])

    start_time = time.time()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    end_time = time.time()

    print(f'Training time: {(end_time - start_time):.1f}s')
    pred = model.predict(x_test)
    test_accuracy = np.mean(kr.metrics.categorical_accuracy(y_test, pred))
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


def create_lstm_model(input_dim, output_dim, embedding_size, lstm_units):
    model = kr.Sequential()
    model.add(kr.layers.Embedding(input_dim=input_dim, mask_zero=True, output_dim=embedding_size))
    model.add(kr.layers.LSTM(units=lstm_units))
    model.add(kr.layers.Dense(output_dim, activation='softmax'))
    model.summary()
    return model


def save_model(model, folder, name, add_timestamp=True):
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_name = name
    if add_timestamp:
        now = datetime.datetime.now()
        timestamp = str(now).replace(':', '-')
        file_name += f'_{timestamp}'

    file_name += '.h5'

    path = os.path.join(folder, file_name)
    model.save(path, overwrite=True, include_optimizer=False, save_format='h5')


def load_model(path):
    return kr.models.load_model(path)


if __name__ == '__main__':
    dataset = get_dataset('../data', depth=2, variables=5, test_size=.1, indexed_encoding=True)
    x_train, x_test, y_train, y_test, input_dictionary, output_dictionary = dataset

    learning_rate = 0.001
    batch_size = 64
    epochs = 70
    embedding_size = 64
    lstm_units = 128

    input_dim = dataset[0].shape[-1]
    output_dim = dataset[2].shape[-1]
    model = create_lstm_model(input_dim, output_dim, embedding_size, lstm_units)
    history, test_accuracy = train_model(model,
                                         dataset,
                                         learning_rate,
                                         batch_size,
                                         epochs)
    plot_loss_accuracy(history)
    save_model(model, '../results', 'lstm')

    model = load_model('../results/lstm_2021-03-11 17-00-45.414892.h5')