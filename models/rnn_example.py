import tensorflow.keras as kr


def create_rnn_model(rnn_cls, num_layers, input_dim, output_dim, embedding_size, hidden_units,
                     bidirectional=False, out_activation='softmax', max_length=None):
    model = kr.Sequential()
    model.add(kr.layers.Embedding(input_dim=input_dim + 1, mask_zero=True, output_dim=embedding_size))

    for i in range(num_layers):
        if i == num_layers - 1:
            if bidirectional:
                model.add(kr.layers.Bidirectional(rnn_cls(units=hidden_units)))
            else:
                model.add(rnn_cls(units=hidden_units))
        else:
            if bidirectional:
                model.add(kr.layers.Bidirectional(rnn_cls(units=hidden_units, return_sequences=True)))
            else:
                model.add(rnn_cls(units=hidden_units, return_sequences=True))

    model.add(kr.layers.Dense(output_dim, activation=out_activation))
    model.summary()
    return model


def create_no_embedding_lstm(num_layers, input_dim, output_dim, embedding_size, hidden_units, bidirectional=False,
                             max_length=None):
    model = kr.Sequential()
    for i in range(num_layers):
        if bidirectional:
            if i == num_layers - 1:
                if i == 0:
                    model.add(kr.layers.Bidirectional(kr.layers.LSTM(units=hidden_units), input_shape=(13, 10)))
                else:
                    model.add(kr.layers.Bidirectional(kr.layers.LSTM(units=hidden_units)))
            else:
                if i == 0:
                    model.add(kr.layers.Bidirectional(kr.layers.LSTM(units=hidden_units, return_sequences=True),
                                                      input_shape=(13, 10)))
                else:
                    model.add(kr.layers.Bidirectional(kr.layers.LSTM(units=hidden_units, return_sequences=True)))
        else:
            if i == num_layers - 1:
                if i == 0:
                    model.add(kr.layers.LSTM(units=hidden_units), input_shape=(13, 10))
                else:
                    model.add(kr.layers.LSTM(units=hidden_units))
            else:
                if i == 0:
                    model.add(kr.layers.LSTM(units=hidden_units, return_sequences=True,
                                                      input_shape=(13, 10)))
                else:
                    model.add(kr.layers.LSTM(units=hidden_units, return_sequences=True))

    model.add(kr.layers.Dense(output_dim, activation='softmax'))
    model.summary()
    return model


def create_lstm_model(num_layers, input_dim, output_dim, embedding_size, hidden_units, bidirectional=False,
                      out_activation='softmax', max_length=None):
    return create_rnn_model(kr.layers.LSTM, num_layers,
                            input_dim, output_dim,
                            embedding_size, hidden_units,
                            bidirectional, out_activation, max_length)


def create_gru_model(num_layers, input_dim, output_dim, embedding_size, hidden_units, bidirectional=False,
                     out_activation='softmax', max_length=None):
    return create_rnn_model(kr.layers.GRU, num_layers,
                            input_dim, output_dim,
                            embedding_size, hidden_units,
                            bidirectional, out_activation, max_length)


def create_simple_rnn_model(num_layers, input_dim, output_dim, embedding_size, hidden_units, bidirectional=False,
                            out_activation='softmax', max_length=None):
    return create_rnn_model(kr.layers.SimpleRNN, num_layers,
                            input_dim, output_dim,
                            embedding_size, hidden_units,
                            bidirectional, out_activation, max_length)
