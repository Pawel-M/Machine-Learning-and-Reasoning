import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
from tqdm import tqdm

import dataset
import dataset.encoding
from dataset.common import get_separated_sequences_mental_models_dataset


def create_varying_inference_model1(num_variables, max_input_length, num_symbols,
                                    batch_size,
                                    embedding_size,
                                    hidden_units,
                                    epochs):
    # Without specific token for start of sequence (index 0) and end of sequence (index 1) - (0,0,0,0,0) equals end
    print('max_input_length', max_input_length)

    # Create input for encoder
    encoder_inputs = kr.Input(shape=(2, max_input_length))

    print('input', encoder_inputs.shape)

    # Make model - Encoder (flatten / concatenate subsentences to one vector (subsentence representation))
    nn_input = kr.Input(shape=(num_variables))
    nn_embedding_layer = kr.layers.Embedding(num_symbols + 1, embedding_size)(encoder_inputs)
    nn_flatten = tf.keras.layers.Reshape((nn_embedding_layer.shape[1], -1))(nn_embedding_layer)
    encoder = kr.layers.LSTM(hidden_units, return_sequences=True, return_state=True, activation='relu')
    encoder_outputs, state_h, state_c = encoder(nn_flatten)
    encoder_states = [state_h, state_c]

    # Create decoder
    decoder_inputs = kr.Input(shape=(None, num_variables))
    decoder = kr.layers.LSTM(hidden_units, return_sequences=True, return_state=True, activation='relu')
    decoder_outputs, _, _ = decoder(decoder_inputs,
                                    initial_state=encoder_states)
    decoder_dense = kr.layers.Dense(num_variables, activation='tanh')
    output = decoder_dense(decoder_outputs)

    ## Define training model
    model_train = kr.Model(inputs=[encoder_inputs, decoder_inputs], outputs=output)
    model_train.summary()

    ## Train model
    model_train.compile(optimizer=kr.optimizers.Adam(learning_rate=1e-3),
                        loss=kr.losses.mse)

    callbacks = [kr.callbacks.EarlyStopping(patience=20, min_delta=1e-5, restore_best_weights=True)]
    history = model_train.fit([ds.x_train, ds.y_train_d], ds.y_train,
                              validation_data=([ds.x_valid, ds.y_valid_d], ds.y_valid),
                              epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    ## Define testing models (no teacher forcing)
    encoder_model = kr.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = kr.Input(shape=(hidden_units,))
    decoder_state_input_c = kr.Input(shape=(hidden_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = kr.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    # Returned trained models, and history of training
    return model_train, history, encoder_model, decoder_model


def create_varying_inference_model2(num_variables, max_input_length, num_symbols,
                                    batch_size,
                                    embedding_size,
                                    hidden_units,
                                    epochs):
    # With specific token for start of sequence (index 0) and end of sequence (index 1)
    print('max_input_length', max_input_length)
    # Create input for encoder
    encoder_inputs = kr.Input(shape=(2, max_input_length))

    print('input', encoder_inputs.shape)

    # Make model - Encoder (flatten / concatenate subsentences to one vector (subsentence representation))
    nn_input = kr.Input(shape=(num_variables))
    nn_embedding_layer = kr.layers.Embedding(num_symbols + 1, embedding_size)(encoder_inputs)
    nn_flatten = tf.keras.layers.Reshape((nn_embedding_layer.shape[1], -1))(nn_embedding_layer)
    encoder = kr.layers.LSTM(hidden_units, return_sequences=True, return_state=True, activation='relu')
    encoder_outputs, state_h, state_c = encoder(nn_flatten)
    encoder_states = [state_h, state_c]

    # Create decoder
    decoder_inputs = kr.Input(shape=(None, num_variables + 2))
    decoder = kr.layers.LSTM(hidden_units, return_sequences=True, return_state=True, activation='relu')
    decoder_outputs, _, _ = decoder(decoder_inputs,
                                    initial_state=encoder_states)
    decoder_dense = kr.layers.Dense(num_variables + 2, activation='tanh')
    output = decoder_dense(decoder_outputs)

    ## Define training model
    model_train = kr.Model(inputs=[encoder_inputs, decoder_inputs], outputs=output)
    model_train.summary()

    ## Train model
    model_train.compile(optimizer=kr.optimizers.Adam(learning_rate=1e-3),
                        loss=kr.losses.mse)

    callbacks = [kr.callbacks.EarlyStopping(patience=20, min_delta=1e-5, restore_best_weights=True)]
    history = model_train.fit([ds.x_train, ds.y_train_d], ds.y_train,
                              validation_data=([ds.x_valid, ds.y_valid_d], ds.y_valid),
                              epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    ## Define testing models (no teacher forcing)
    encoder_model = kr.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = kr.Input(shape=(hidden_units,))
    decoder_state_input_c = kr.Input(shape=(hidden_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = kr.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    # Returned trained models, and history of training
    return model_train, history, encoder_model, decoder_model


def create_varying_inference_model3(num_variables, max_input_length, num_symbols,
                                    batch_size,
                                    embedding_size,
                                    hidden_units,
                                    epochs):
    # Without specific token for start of sequence (index 0) and end of sequence (index 1) - (0,0,0,0,0) equals end
    # Based on characters instead of subsentences
    print('max_input_length', max_input_length)

    # Create input for encoder
    encoder_inputs = kr.Input(shape=(11, max_input_length))

    print('input', encoder_inputs.shape)

    # Make model - Encoder (flatten / concatenate subsentences to one vector (subsentence representation))
    nn_input = kr.Input(shape=(1))
    print('input2', nn_input.shape)
    nn_embedding_layer = kr.layers.Embedding(num_symbols + 1, embedding_size)(encoder_inputs)
    nn_flatten = tf.keras.layers.Reshape((nn_embedding_layer.shape[1], -1))(nn_embedding_layer)
    print('embedding_layer', nn_embedding_layer.shape)
    encoder = kr.layers.LSTM(hidden_units, return_sequences=True, return_state=True, activation='relu')
    encoder_outputs, state_h, state_c = encoder(nn_flatten)
    encoder_states = [state_h, state_c]

    # Create decoder
    decoder_inputs = kr.Input(shape=(None, num_variables))
    decoder = kr.layers.LSTM(hidden_units, return_sequences=True, return_state=True, activation='relu')
    decoder_outputs, _, _ = decoder(decoder_inputs,
                                    initial_state=encoder_states)
    decoder_dense = kr.layers.Dense(num_variables, activation='tanh')
    output = decoder_dense(decoder_outputs)

    ## Define training model
    model_train = kr.Model(inputs=[encoder_inputs, decoder_inputs], outputs=output)
    model_train.summary()

    ## Train model
    model_train.compile(optimizer=kr.optimizers.Adam(learning_rate=1e-3),
                        loss=kr.losses.mse)

    callbacks = [kr.callbacks.EarlyStopping(patience=20, min_delta=1e-5, restore_best_weights=True)]
    history = model_train.fit([ds.x_train, ds.y_train_d], ds.y_train,
                              validation_data=([ds.x_valid, ds.y_valid_d], ds.y_valid),
                              epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    ## Define testing models (no teacher forcing)
    encoder_model = kr.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = kr.Input(shape=(hidden_units,))
    decoder_state_input_c = kr.Input(shape=(hidden_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = kr.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    # Returned trained models, and history of training
    return model_train, history, encoder_model, decoder_model


def decode_sequence1(input_seq, encoder_model, decoder_model, num_variables):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_variables))
    # Populate the first character of target sequence with the start character.
    #     target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_output = target_seq
    while not stop_condition:
        output, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Save MMs
        pred = np.rint(output).astype(int)

        # Exit condition: hit max length
        # this padding, such that all arrays have the same size in decoded_output.
        if np.sum(pred) == 0:
            stop_condition = True
        else:
            decoded_output = np.concatenate((decoded_output, pred), axis=1)

        # Update the target sequence (of length 1).
        target_seq = pred

        # Update states
        states_value = [h, c]

    return decoded_output[:, 1:, :]


def decode_sequence2(input_seq, encoder_model, decoder_model, num_variables):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_variables + 2))
    # Populate the first character of target sequence with the start character.
    #     target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_output = target_seq
    while not stop_condition:
        output, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Save MMs
        pred = np.rint(output).astype(int)

        # Exit condition: hit max length
        # this padding, such that all arrays have the same size in decoded_output.
        if pred[0, 0, 1] == 1 and np.sum(pred) == 1:
            stop_condition = True
        else:
            decoded_output = np.concatenate((decoded_output, pred), axis=1)

        # Update the target sequence (of length 1).
        target_seq = pred

        # Update states
        states_value = [h, c]

    return decoded_output[:, 1:, :]


def decode_sequences1(data, encoder_model, decoder_model, num_variables):
    preds = []
    for i in tqdm(range(0, ds.x_test.shape[0])):
        pred = decode_sequence1(data[[i]], encoder_model, decoder_model, num_variables)
        preds.append(pred)

    return preds


def decode_sequences2(data, encoder_model, decoder_model, num_variables):
    preds = []
    for i in tqdm(range(0, ds.x_test.shape[0])):
        pred = decode_sequence2(data[[i]], encoder_model, decoder_model, num_variables)
        preds.append(pred)

    return preds


def two_way_mse(y_true, y_pred):
    y_true_float = tf.cast(y_true, y_pred.dtype)
    diff = (y_true_float - y_pred) ** 2
    print(diff)
    return tf.reduce_mean(diff)


def show_subsentence_inference(model, ds, decoding_dictionary, idxs):
    sub_model = model.layers[2]
    for i in idxs:
        for j in range(2):
            x = ds.x_test[i][j]
            pred = sub_model.predict(x[np.newaxis, ...])
            print(dataset.encoding.decode_sentence(x, decoding_dictionary, ds.indexed_encoding))
            print(np.rint(pred))


def add_zero_row1(data, position):
    if position == 'front':
        temp = np.zeros((data.shape[0], data.shape[1] + 1, data.shape[2]))
        temp[:, 1:, :] = data
    elif position == 'last':
        temp = np.zeros((data.shape[0], data.shape[1] + 1, data.shape[2]))
        temp[:, :-1, :] = data

    return temp


def add_zero_row2(dst, position, num_variables):
    start_vec = np.array([1, 0] + [0] * (num_variables))
    end_vec = np.array([0, 1] + [0] * (num_variables))
    if position == 'front':
        data = np.zeros((dst.shape[0], dst.shape[1] + 1, dst.shape[2] + 2))
        data[:, 1:, 2:] = dst
        for i in range(data.shape[0]):
            data[i][data[i].sum(axis=1) == 0] = end_vec
            data[i][0, :] = start_vec
    elif position == 'last':
        data = np.zeros((dst.shape[0], dst.shape[1] + 1, dst.shape[2] + 2))
        data[:, :-1, 2:] = dst
        for i in range(data.shape[0]):
            data[i][data[i].sum(axis=1) == 0] = end_vec
            data[i][-1, :] = end_vec

    return data


def concat_subsentences(data):
    temp = np.array(data[0][0].tolist() + [10] + data[0][1].tolist())[np.newaxis, ...]
    for i in range(1, data.shape[0]):
        sentence = np.array(data[i][0].tolist() + [10] + data[i][1].tolist())[np.newaxis, ...]
        temp = np.concatenate((temp, sentence), axis=0)

    return temp[..., np.newaxis]


def same_MMs(true, pred):
    true = true.tolist()
    pred = pred.tolist()
    # Check if all occur
    for i in range(len(pred)):
        if pred[i] in true:
            # If occur, remove from true
            true.remove(pred[i])

    # If nothing in true, everything is predicted correctly
    return true == []


def remove_zero_rows1(data):
    return data[np.sum(data, axis=1) != 0]


def remove_zero_rows2(data):
    return data[np.logical_not(np.logical_and(data[:, 1] == 1, np.sum(data, axis=1) == 1))]


def train_encoder_decoder(type, start_index, ds,
                          num_variables,
                          batch_size,
                          embedding_size,
                          hidden_units,
                          epochs):
    dec_in, dec_out = dataset.encoding.create_decoding_dictionaries(ds.input_dictionary, ds.output_dictionary)

    if not start_index:
        ds.y_train_d = add_zero_row1(ds.y_train, 'front')
        ds.y_train = add_zero_row1(ds.y_train, 'last')
        ds.y_valid_d = add_zero_row1(ds.y_valid, 'front')
        ds.y_valid = add_zero_row1(ds.y_valid, 'last')
        ds.y_test_d = add_zero_row1(ds.y_test, 'front')
        ds.y_test = add_zero_row1(ds.y_test, 'last')
    else:
        ds.y_train_d = add_zero_row2(ds.y_train, 'front', num_variables)
        ds.y_train = add_zero_row2(ds.y_train, 'last', num_variables)
        ds.y_valid_d = add_zero_row2(ds.y_valid, 'front', num_variables)
        ds.y_valid = add_zero_row2(ds.y_valid, 'last', num_variables)
        ds.y_test_d = add_zero_row2(ds.y_test, 'front', num_variables)
        ds.y_test = add_zero_row2(ds.y_test, 'last', num_variables)

    if type == 'symbol':
        ds.x_train = concat_subsentences(ds.x_train)
        ds.x_valid = concat_subsentences(ds.x_valid)
        ds.x_test = concat_subsentences(ds.x_test)

    # ds.x_test = ds.x_test[:30]
    # ds.y_test = ds.y_test[:30]

    num_variables = 5
    num_operators = 5  # and, or, not
    num_symbols = num_variables + num_operators
    max_input_length = ds.x_train.shape[-1]

    # 1: subsentences no index, 2: subsentences index, 3: symbols no index
    if type == 'subsentence' and not start_index:
        model_train, history, encoder_model, decoder_model = create_varying_inference_model1(num_variables,
                                                                                             max_input_length,
                                                                                             num_symbols,
                                                                                             batch_size,
                                                                                             embedding_size,
                                                                                             hidden_units,
                                                                                             epochs)
    elif type == 'subsentence' and start_index:
        model_train, history, encoder_model, decoder_model = create_varying_inference_model2(num_variables,
                                                                                             max_input_length,
                                                                                             num_symbols,
                                                                                             batch_size,
                                                                                             embedding_size,
                                                                                             hidden_units,
                                                                                             epochs)
    elif type == 'symbol' and not start_index:
        model_train, history, encoder_model, decoder_model = create_varying_inference_model3(num_variables,
                                                                                             max_input_length,
                                                                                             num_symbols,
                                                                                             batch_size,
                                                                                             embedding_size,
                                                                                             hidden_units,
                                                                                             epochs)
    else:
        raise NotImplementedError()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(range(len(loss)), loss, label='loss')
    plt.plot(range(len(val_loss)), loss, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    if not start_index:
        preds = decode_sequences1(ds.x_test, encoder_model, decoder_model, num_variables)
    else:
        preds = decode_sequences2(ds.x_test, encoder_model, decoder_model, num_variables)

    print('errors:')
    errors = 0
    for i in range(len(preds)):
        if not start_index:
            y_preproc = remove_zero_rows1(ds.y_test[i])
        else:
            y_preproc = remove_zero_rows2(ds.y_test[i])
        if same_MMs(y_preproc, preds[i][0]):
            continue
        print(dataset.encoding.decode_sentence(ds.x_test[i][0], dec_in, ds.indexed_encoding))
        print(dataset.encoding.decode_sentence(ds.x_test[i][1], dec_in, ds.indexed_encoding))
        print('TRUE: ', y_preproc)
        print('PREDICTED: ', preds[i], '\n')

        errors += 1

    accuracy = 1 - float(errors) / ds.x_test.shape[0]
    print('errors', int(errors))
    print(f'accuracy: {accuracy * 100:.1f}%')

    return accuracy


if __name__ == '__main__':
    ds = get_separated_sequences_mental_models_dataset('../data', 'encoded_and_trees_single_mms_type_I',
                                                       num_variables=5, max_depth=2,
                                                       test_size=.1, valid_size=.1,
                                                       indexed_encoding=True, pad_mental_models=True)

    type = 'subsentence'  # Could be "subsentence" or "symbol"
    start_index = True  # True or false (True -> specific start and end index for Start / End of sequence)
    num_variables = 5
    batch_size = 8
    embedding_size = 10
    hidden_units = 128
    epochs = 1000
    acc = train_encoder_decoder(type, start_index, ds,
                                num_variables,
                                batch_size,
                                embedding_size,
                                hidden_units,
                                epochs)
    print(acc)
