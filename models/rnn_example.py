import datetime
import os

import tensorflow.keras as kr


def create_rnn_model(rnn_cls, num_layers, input_dim, output_dim, embedding_size, hidden_units, bidirectional=False):
    model = kr.Sequential()
    model.add(kr.layers.Embedding(input_dim=input_dim, mask_zero=True, output_dim=embedding_size))

    for i in range(num_layers):
        if bidirectional:
            model.add(kr.layers.Bidirectional(rnn_cls(units=hidden_units)))
        else:
            model.add(rnn_cls(units=hidden_units))

    model.add(kr.layers.Dense(output_dim, activation='softmax'))
    model.summary()
    return model


def create_lstm_model(num_layers, input_dim, output_dim, embedding_size, hidden_units, bidirectional=False):
    return create_rnn_model(kr.layers.LSTM, num_layers,
                            input_dim, output_dim,
                            embedding_size, hidden_units,
                            bidirectional)


def create_gru_model(num_layers, input_dim, output_dim, embedding_size, hidden_units, bidirectional=False):
    return create_rnn_model(kr.layers.GRU, num_layers,
                            input_dim, output_dim,
                            embedding_size, hidden_units,
                            bidirectional)


def create_simple_rnn_model(num_layers, input_dim, output_dim, embedding_size, hidden_units, bidirectional=False):
    return create_rnn_model(kr.layers.SimpleRNN, num_layers,
                            input_dim, output_dim,
                            embedding_size, hidden_units,
                            bidirectional)


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
