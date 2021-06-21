from dataset.common import get_separated_sequences_mental_models_dataset
from models.mm_inference import MultiMMInferenceLayer
import tensorflow.keras as kr
from tqdm import tqdm

# warnings.filterwarnings("ignore")
import tensorflow as tf
# import keras.backend as kb
import numpy as np


def create_multi_mm_inference_model(input,
                                    hidden_units,
                                    embedding_size,
                                    max_sub_mental_models,
                                    mm_l1,
                                    score_l1,
                                    max_length,
                                    input_dim,
                                    output_dim):
    print('max_input_length', max_length)
    # input = kr.Input(shape=(2, max_length))
    split_layer = kr.layers.Lambda(lambda x: (x[:, 0], x[:, 1]))(input)

    nn_input = kr.Input(max_length)
    nn_embedding_layer = kr.layers.Embedding(input_dim + 1, embedding_size)(nn_input)
    flatten_layer = kr.layers.Flatten()(nn_embedding_layer)
    nn_hidden = kr.layers.Dense(hidden_units, activation='relu')(flatten_layer)
    nn_output = kr.layers.Dense(output_dim * max_sub_mental_models,
                                activation='tanh',
                                activity_regularizer=kr.regularizers.L1(mm_l1))(nn_hidden)
    score_output = kr.layers.Dense(max_sub_mental_models,
                                   activation='sigmoid',
                                   activity_regularizer=kr.regularizers.L1(score_l1))(nn_hidden)
    nn_reshape = kr.layers.Reshape((max_sub_mental_models, output_dim))(nn_output)
    sub_sequence_nn = kr.Model(inputs=nn_input, outputs=[nn_reshape, score_output], name='sub-sequence-NN')
    sub_sequence_nn.summary()

    mms_and_scores = sub_sequence_nn(split_layer[0]), sub_sequence_nn(split_layer[1])
    mm_inference_layer = MultiMMInferenceLayer()(mms_and_scores)

    return mm_inference_layer


class CombineMentalModelsLayer(kr.layers.Layer):
    def call(self, inputs, **kwargs):
        mms, scores, scores_prime = inputs
        result = tf.expand_dims(mms * tf.expand_dims(scores, axis=-1), axis=-3)
        combined = tf.reduce_sum(result * tf.expand_dims(scores_prime, axis=-1), axis=-2)
        # normalization = tf.reduce_sum(tf.expand_dims(scores, axis=-2) * scores_prime, axis=-1, keepdims=True)
        # normalized = combined / tf.maximum(1.0, normalization)
        return combined


def create_varying_inference_model2(epochs,
                                    batch_size,
                                    embedding_size,
                                    encoder_hidden_units,
                                    max_sub_mental_models,
                                    mm_l1,
                                    score_l1,
                                    num_operators,  # and, or, not,
                                    max_length,
                                    output_dim,
                                    num_variables,
                                    max_input_length,
                                    ds):
    input_dim = num_variables + num_operators
    print('max_input_length', max_input_length)

    encoder_inputs = kr.Input(shape=(2, max_length), name='Encoder Input')
    encoder_mms, encoder_scores = create_multi_mm_inference_model(encoder_inputs,
                                                                  encoder_hidden_units,
                                                                  embedding_size,
                                                                  max_sub_mental_models,
                                                                  mm_l1,
                                                                  score_l1,
                                                                  max_length,
                                                                  input_dim,
                                                                  output_dim)

    # encoder_mms, encoder_scores = encoder(encoder_inputs)
    # tf.print(encoder_mms)
    print(encoder_mms)

    nn_flatten = tf.keras.layers.Reshape((encoder_mms.shape[-2] * encoder_mms.shape[-1],))
    nn_flatten_output = nn_flatten(encoder_mms)
    nn_concat = tf.keras.layers.Concatenate(axis=1)
    nn_concat_output = nn_concat([nn_flatten_output, encoder_scores])

    # mm = 9 x 5
    # scores = 9 x 1

    # scores_prime = m x 9
    # [1, 0, 0, 1, ...]

    # Create decoder
    decoder_states = [nn_concat_output, nn_concat_output]

    lstm_dim = max_sub_mental_models ** 2 * (num_variables + 1)

    decoder_inputs = kr.Input(shape=(None, num_variables), name='Decoder Inputs (MMs)')
    decoder = kr.layers.LSTM(lstm_dim, return_sequences=True, return_state=True, activation='relu')
    decoder_outputs, _, _ = decoder(decoder_inputs,
                                    initial_state=decoder_states)
    decoder_dense = kr.layers.Dense(encoder_scores.shape[-1], activation='sigmoid')
    output = decoder_dense(decoder_outputs)

    combine_layer = CombineMentalModelsLayer()
    combined_output = combine_layer([encoder_mms, encoder_scores, output])

    ## Define training model
    model_train = kr.Model(inputs=[encoder_inputs, decoder_inputs], outputs=combined_output)
    model_train.summary()

    ## Train model
    model_train.compile(optimizer=kr.optimizers.Adam(learning_rate=1e-3),
                        loss=kr.losses.mse)

    callbacks = [kr.callbacks.EarlyStopping(patience=20, min_delta=1e-5, restore_best_weights=True)]
    history = model_train.fit([ds.x_train, ds.y_train_d], ds.y_train,
                              validation_data=([ds.x_valid, ds.y_valid_d], ds.y_valid),
                              epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    # model_pred = kr.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[combined_output, output])
    # model_pred.summary()

    ## Define testing models (no teacher forcing)
    encoder_model = kr.Model(encoder_inputs, [encoder_mms, encoder_scores, nn_concat_output])

    decoder_state_input_h = kr.Input(shape=(lstm_dim,))
    decoder_state_input_c = kr.Input(shape=(lstm_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_input_mms = kr.Input(shape=(max_sub_mental_models ** 2, num_variables))
    decoder_input_scores = kr.Input(shape=(max_sub_mental_models ** 2,))
    decoder_mm_inputs = [decoder_input_mms, decoder_input_scores]

    decoder_outputs, state_h, state_c = decoder(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    comb_output = combine_layer([decoder_input_mms, decoder_input_scores, decoder_outputs])
    decoder_model = kr.Model(
        [decoder_inputs] + decoder_states_inputs + decoder_mm_inputs,
        [comb_output] + decoder_states)

    # Returned trained models, and history of training
    return model_train, history, encoder_model, decoder_model


def decode_sequence2(input_seq, encoder_model, decoder_model, num_variables):
    # Encode the input as state vectors.
    encoder_mms, encoder_scores, nn_concat_output = encoder_model.predict(input_seq)

    states_value = [nn_concat_output, nn_concat_output]

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_variables))
    # Populate the first character of target sequence with the start character.
    #     target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_output = target_seq
    while not stop_condition:
        dec_output, h, c = decoder_model.predict(
            [target_seq] + states_value + [encoder_mms, encoder_scores])

        # Save MMs
        pred = np.rint(dec_output).astype(int)

        # Exit condition: hit max length
        # this padding, such that all arrays have the same size in decoded_output.
        # if decoded_output.shape[1] > num_variables:
        if np.sum(np.abs(pred)) == 0:
            stop_condition = True
        else:
            decoded_output = np.concatenate((decoded_output, pred), axis=1)

        # Update the target sequence (of length 1).
        target_seq = pred

        # Update states
        states_value = [h, c]

    return decoded_output[:, 1:, :]


def train_multi_mms_model2(ds,
                           epochs, batch_size,
                           embedding_size, encoder_hidden_units,
                           max_sub_mental_models,
                           mm_l1,
                           score_l1):
    dec_in, dec_out = dataset.encoding.create_decoding_dictionaries(ds.input_dictionary, ds.output_dictionary)

    ds.y_train_d = add_zero_row(ds.y_train, 'front')
    ds.y_train = add_zero_row(ds.y_train, 'last')
    ds.y_valid_d = add_zero_row(ds.y_valid, 'front')
    ds.y_valid = add_zero_row(ds.y_valid, 'last')
    ds.y_test_d = add_zero_row(ds.y_test, 'front')
    ds.y_test = add_zero_row(ds.y_test, 'last')

    num_variables = 5
    max_input_length = ds.x_train.shape[-1]

    num_operators = 5  # and, or, not
    max_length = 5
    output_dim = 5

    model_train, history, encoder_model, decoder_model = create_varying_inference_model2(epochs,
                                                                                         batch_size,
                                                                                         embedding_size,
                                                                                         encoder_hidden_units,
                                                                                         max_sub_mental_models,
                                                                                         mm_l1,
                                                                                         score_l1,
                                                                                         num_operators,
                                                                                         # and, or, not,
                                                                                         max_length,
                                                                                         output_dim,
                                                                                         num_variables,
                                                                                         max_input_length,
                                                                                         ds)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(range(len(loss)), loss, label='loss')
    plt.plot(range(len(val_loss)), loss, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    preds = []
    for i in tqdm(range(0, ds.x_test.shape[0])):
        pred = decode_sequence2(ds.x_test[[i]], encoder_model, decoder_model, num_variables)
        preds.append(pred)

    print('predictions:')
    for i in range(len(preds)):
        y_preproc = remove_zero_rows(ds.y_test[i])
        print('TRUE: ', y_preproc)
        print('PREDICTED: ', preds[i], '\n')

    print('errors:')
    errors = 0
    for i in range(len(preds)):
        y_preproc = remove_zero_rows(ds.y_test[i])
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


def add_zero_row(data, position):
    if position == 'front':
        temp = np.zeros((data.shape[0], data.shape[1] + 1, data.shape[2]))
        temp[:, 1:, :] = data
    elif position == 'last':
        temp = np.zeros((data.shape[0], data.shape[1] + 1, data.shape[2]))
        temp[:, :-1, :] = data

    return temp


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


import dataset
import matplotlib.pyplot as plt


def remove_zero_rows(data):
    return data[np.sum(np.abs(data), axis=1) != 0]


def train_multiple_times2(n, ds,
                         epochs, batch_size,
                         embedding_size, encoder_hidden_units,
                         max_sub_mental_models,
                         mm_l1,
                         score_l1):
    accuracies_str = ''
    accuracies = []
    for i in range(n):
        accuracy = train_multi_mms_model2(ds,
                                         epochs, batch_size,
                                         embedding_size, encoder_hidden_units,
                                         max_sub_mental_models,
                                         mm_l1,
                                         score_l1)
        accuracies.append(accuracy)
        accuracies_str += f'{accuracy} '

    print('Resulting accuracies:')
    print(accuracies_str)
    return accuracies


if __name__ == '__main__':
    ds = get_separated_sequences_mental_models_dataset('./data', 'encoded_and_trees_multiple_mms',
                                                       num_variables=5, max_depth=2,
                                                       test_size=.1, valid_size=.1,
                                                       indexed_encoding=True, pad_mental_models=True)

    # ds.x_test = ds.x_test[:30, ...]
    # ds.y_test = ds.y_test[:30, ...]
    n = 6
    epochs = 300
    batch_size = 8
    embedding_size = 10
    encoder_hidden_units = 128
    max_sub_mental_models = 3
    mm_l1 = 0.0
    score_l1 = 0.0

    accuracies = train_multiple_times2(n, ds,
                                        epochs, batch_size,
                                        embedding_size, encoder_hidden_units,
                                        max_sub_mental_models,
                                        mm_l1,
                                        score_l1)
