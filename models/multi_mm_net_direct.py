# import keras.backend as kb
import numpy as np
# warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow.keras as kr
from tqdm import tqdm

from dataset.common import get_separated_sequences_mental_models_dataset
from models.single_mm_net import MultiMMInferenceLayer


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

    model = kr.Model(inputs=input, outputs=mm_inference_layer)
    model.summary()

    return model


def create_multi_mm_net_direct(epochs,
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
    # Without specific token for start of sequence (index 0) and end of sequence (index 1) - (0,0,0,0,0) equals end
    # Based on characters instead of subsentences
    # Initialise parameters

    input_dim = num_variables + num_operators
    print('max_input_length', max_input_length)

    encoder_inputs = kr.Input(shape=(2, max_length))
    encoder = create_multi_mm_inference_model(encoder_inputs,
                                              encoder_hidden_units,
                                              embedding_size,
                                              max_sub_mental_models,
                                              mm_l1,
                                              score_l1,
                                              max_length,
                                              input_dim,
                                              output_dim)

    encoder_mms, encoder_scores = encoder(encoder_inputs)
    # tf.print(encoder_mms)
    print(encoder_mms)
    nn_flatten = tf.keras.layers.Reshape((encoder_mms.shape[-2] * encoder_mms.shape[-1],))(encoder_mms)
    nn_concat = tf.keras.layers.Concatenate(axis=1)([nn_flatten, encoder_scores])
    encoder_states = [nn_concat, nn_concat]

    # Create decoder
    decoder_inputs = kr.Input(shape=(None, num_variables))
    decoder = kr.layers.LSTM(54, return_sequences=True, return_state=True, activation='relu')
    decoder_outputs, _, _ = decoder(decoder_inputs,
                                    initial_state=encoder_states)
    decoder_dense = kr.layers.Dense(num_variables, activation='tanh')
    output = decoder_dense(decoder_outputs)

    ## Define training model
    model_train = kr.Model(inputs=[encoder_inputs, decoder_inputs], outputs=output)
    model_train.summary()

    ## Train model
    model_train.compile(optimizer=kr.optimizers.Adam(learning_rate=1e-3),
                        loss=kr.losses.mse)  # loss=custom_loss)  # loss=kr.losses.mse)

    callbacks = [kr.callbacks.EarlyStopping(patience=20, min_delta=1e-5, restore_best_weights=True)]
    history = model_train.fit([ds.x_train, ds.y_train_d], ds.y_train,
                              validation_data=([ds.x_valid, ds.y_valid_d], ds.y_valid),
                              epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    ## Define testing models (no teacher forcing)
    encoder_model = kr.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = kr.Input(shape=(54,))
    decoder_state_input_c = kr.Input(shape=(54,))
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


def add_zero_row(data, position):
    if position == 'front':
        temp = np.zeros((data.shape[0], data.shape[1] + 1, data.shape[2]))
        temp[:, 1:, :] = data
    elif position == 'last':
        temp = np.zeros((data.shape[0], data.shape[1] + 1, data.shape[2]))
        temp[:, :-1, :] = data

    return temp


def decode_sequence(input_seq, encoder_model, decoder_model, num_variables):
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
        # if decoded_output.shape[1] > num_variables:
        if np.sum(np.abs(pred)) == 0 or decoded_output.shape[1] > num_variables:
            stop_condition = True
        else:
            decoded_output = np.concatenate((decoded_output, pred), axis=1)

        # Update the target sequence (of length 1).
        target_seq = pred

        # Update states
        states_value = [h, c]

    return decoded_output[:, 1:, :]


def decode_sequences(data, encoder_model, decoder_model, num_variables):
    preds = []
    for i in tqdm(range(0, data.shape[0])):
        pred = decode_sequence(data[[i]], encoder_model, decoder_model, num_variables)
        preds.append(pred)

    return preds


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
    return data[np.sum(data, axis=1) != 0]


def train_multi_mms_model(ds,
                          epochs, batch_size,
                          embedding_size, encoder_hidden_units,
                          max_sub_mental_models,
                          mm_l1,
                          score_l1,
                          return_models=False):
    dec_in, dec_out = dataset.encoding.create_decoding_dictionaries(ds.input_dictionary, ds.output_dictionary)

    ds.y_train_d = add_zero_row(ds.y_train, 'front')
    ds.y_train = add_zero_row(ds.y_train, 'last')
    ds.y_valid_d = add_zero_row(ds.y_valid, 'front')
    ds.y_valid = add_zero_row(ds.y_valid, 'last')
    ds.y_test_d = add_zero_row(ds.y_test, 'front')
    ds.y_test = add_zero_row(ds.y_test, 'last')

    num_variables = 5
    max_input_length = ds.x_train.shape[-1]

    # epochs = 2
    # batch_size = 8
    # embedding_size = 10
    # encoder_hidden_units = 128
    # max_sub_mental_models = 3
    # mm_l1 = 0.0
    # score_l1 = 0.0
    num_operators = 5  # and, or, not
    max_length = 5
    output_dim = 5

    # 1: subsentences no index, 2: subsentences index, 3: symbols no index
    model_train, history, encoder_model, decoder_model = create_multi_mm_net_direct(epochs,
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
                                                                                    ds)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(range(len(loss)), loss, label='loss')
    plt.plot(range(len(val_loss)), loss, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    preds = decode_sequences(ds.x_test, encoder_model, decoder_model, num_variables)
    # for i in range(len(preds)):
    #     y_preproc = remove_zero_rows(ds.y_test[i])
    #     print('TRUE: ', y_preproc)
    #     print('PREDICTED: ', preds[i], '\n')

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

    if return_models:
        return accuracy, model_train, encoder_model, decoder_model
    else:
        return accuracy


def train_multi_mms_model_random(ds,
                                 epochs, batch_size,
                                 embedding_size, encoder_hidden_units,
                                 max_sub_mental_models,
                                 mm_l1,
                                 score_l1):
    dec_in, dec_out = dataset.encoding.create_decoding_dictionaries(ds.input_dictionary, ds.output_dictionary)

    ds.y_train = randomize_MMs(ds.y_train)
    ds.y_train_d = add_zero_row(ds.y_train, 'front')
    ds.y_train = add_zero_row(ds.y_train, 'last')
    ds.y_valid_d = add_zero_row(ds.y_valid, 'front')
    ds.y_valid = add_zero_row(ds.y_valid, 'last')
    ds.y_test_d = add_zero_row(ds.y_test, 'front')
    ds.y_test = add_zero_row(ds.y_test, 'last')

    num_variables = 5
    max_input_length = ds.x_train.shape[-1]

    # epochs = 2
    # batch_size = 8
    # embedding_size = 10
    # encoder_hidden_units = 128
    # max_sub_mental_models = 3
    # mm_l1 = 0.0
    # score_l1 = 0.0
    num_operators = 5  # and, or, not
    max_length = 5
    output_dim = 5

    # 1: subsentences no index, 2: subsentences index, 3: symbols no index
    model_train, history, encoder_model, decoder_model = create_multi_mm_net_direct(epochs,
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
                                                                                    ds)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(range(len(loss)), loss, label='loss')
    plt.plot(range(len(val_loss)), loss, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    preds = decode_sequences(ds.x_test, encoder_model, decoder_model, num_variables)
    # for i in range(len(preds)):
    #     y_preproc = remove_zero_rows(ds.y_test[i])
    #     print('TRUE: ', y_preproc)
    #     print('PREDICTED: ', preds[i], '\n')

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


def train_multiple_times(n, ds,
                         epochs, batch_size,
                         embedding_size, encoder_hidden_units,
                         max_sub_mental_models,
                         mm_l1,
                         score_l1):
    accuracies_str = ''
    accuracies = []
    for i in range(n):
        accuracy = train_multi_mms_model(ds,
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


def randomize_MMs(datas):
    # Get size
    n = datas.shape[1]
    # For every set of MMS
    for i in range(datas.shape[0]):
        data = datas[i]

        # Get indices with MMs (because of padding)
        indxs = np.array(range(n))[np.sum(data, axis=1) != 0]

        # If it has a MM, randomize the MMs
        if indxs.size > 0:
            rand_indxs = np.random.choice(indxs, size=indxs.size, replace=False)

            # Add to new set
            new_indx = np.concatenate([rand_indxs, np.arange(np.max(indxs) + 1, n)])
            datas[i] = data[new_indx, :]

    # Return randomized training data
    return datas


def getitems_by_indices(values, indices):
    return tf.gather(values, indices)


def sort_MMs(a):
    for i in range(a.shape[1] - 1, -1, -1):
        #     if i == num_variables - 1:
        #         # a = tf.argsort(a[:, i], direction='ASCENDING')
        #         a = a[(a[:, i]).argsort()]
        #     else:
        #         a = a[(-a[:, i]).argsort(kind='mergesort')]

        # print("Sort", tf.argsort(a[..., i], direction='ASCENDING'))
        # print("a", a)
        # a = a[tf.argsort(a[:, i], direction='ASCENDING'), :]
        a = getitems_by_indices(a, tf.argsort(a[:, i], direction='ASCENDING'))

    return a


def sort_multiple_MMs(data):
    return tf.map_fn(sort_MMs, data)


import tensorflow.keras.backend as kb
import itertools


def custom_loss(y_actual, y_pred):
    y_actual = sort_multiple_MMs(y_actual)
    y_pred = sort_multiple_MMs(y_pred)
    print(kb.mean(kb.square(y_pred - y_actual), axis=-1))
    return kb.mean(kb.square(y_pred - y_actual), axis=-1)


def custom_loss3(y_actual, y_pred):
    errors = 0.0
    shape = tf.shape(y_actual)
    for l in range(shape[0]):
        error = []
        for perm in itertools.permutations(range(5)):
            for i, p in enumerate(perm):
                # error.append(kb.mean(kb.square(y_actual[l,p,:] - y_pred[l,i,:])))
                errors += kb.mean(kb.square(y_actual[l, p, :] - y_pred[l, i, :]), axis=-1)
        # errors += error

    print(errors)
    # value = kb.mean(kb.concatenate(errors), axis=-1)

    return errors

    # return K.mean(K.square(y_pred - y_true), axis=-1)


def custom_loss1(y_actual, y_pred):
    errors = []
    for i in range(y_actual.shape[1]):
        # max_zero = kb.sum(kb.sum(y_actual, axis=1) == 0)
        # print(max_zero)
        error = []
        true = y_actual[:, i, :]
        for j in range(y_pred.shape[1]):
            pred = y_pred[:, j, :]
            error.append(true - pred)

        errors.append(kb.min(error, axis=1))

    value = kb.mean(kb.square(errors), axis=-1)
    # with a for loop, take min?
    return value

    # return K.mean(K.square(y_pred - y_true), axis=-1)


def custom_loss2(y_actual, y_pred):
    errors = []
    for i in range(y_actual.shape[1]):
        error = []
        true = y_actual[:, i, :]
        if kb.sum(true) == 0:
            for j in range(y_pred.shape[1]):
                pred = y_pred[:, j, :]
                error.append(true - pred)

        errors.append(kb.min(error, axis=1))

    value = kb.mean(kb.square(errors), axis=-1)
    # with a for loop, take min?
    return value


def show_subsentence_inference(model, ds, decoding_dictionary, idxs):
    sub_model = model.layers[1].layers[2]
    for i in idxs:
        xs = ds.x_test[i]
        preds = model.predict(xs[np.newaxis, ...])
        for j in range(2):
            x = ds.x_test[i][j]
            pred = sub_model.predict(x[np.newaxis, ...])
            print(dataset.encoding.decode_sentence(x, decoding_dictionary, ds.indexed_encoding))
            if type(pred) is list:
                print(np.rint(pred[0]), np.rint(pred[1]))
            else:
                print(np.rint(pred))
            print(pred)
        print('combined prediction:')
        print(np.rint(preds[0][0, :45].reshape((9, 5))), np.rint(preds[0][0, 45:]))
        print(preds)
        print()


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    ds = get_separated_sequences_mental_models_dataset('./data', 'encoded_and_trees_multiple_mms',
                                                       num_variables=5, max_depth=2,
                                                       test_size=.1, valid_size=.1,
                                                       indexed_encoding=True, pad_mental_models=True)
    dec_in, dec_out = dataset.encoding.create_decoding_dictionaries(ds.input_dictionary, ds.output_dictionary)

    n = 3
    epochs = 300
    batch_size = 8  # 8 16 32
    embedding_size = 10  # 10 40
    encoder_hidden_units = 512  # 128 256 512 1024
    max_sub_mental_models = 3
    mm_l1 = 0.0
    score_l1 = 0.0

    # accuracies = train_multiple_times(n, ds,
    #                                   epochs, batch_size,
    #                                   embedding_size, encoder_hidden_units,
    #                                   max_sub_mental_models,
    #                                   mm_l1,
    #                                   score_l1)

    accuracy, model_train, encoder_model, decoder_model = train_multi_mms_model(ds,
                                                                                epochs, batch_size,
                                                                                embedding_size, encoder_hidden_units,
                                                                                max_sub_mental_models,
                                                                                mm_l1,
                                                                                score_l1, return_models=True)

    show_subsentence_inference(encoder_model, ds, dec_in, range(ds.x_test.shape[0]))

    print(accuracy)
