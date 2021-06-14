# def find_common_models(set_a, set_b):
#     # the function simply returns all states that are possible in both mental models
#     return set_a.intersection(set_b)

# Model 1 represents the sentence (A or B).
# This sentence can imply three different states:
# 1) A = True, B = False
# 2) A = False, B = True
# 3) A = True, B = True
# Representing these states in vectors:
# 1) (1, 0)
# 2) (0, 1)
# 3) (1, 1)

# Here, mm1 is the vector-representation of the sentence (A or B)
# Here, mm2 is the vector-representation of the sentence (A or not B)

# mm1 = {(1, 0), (0, 1), (1, 1)}
# mm2 = {(1, 0), (1, 1), (0, 0)}
# common_models = find_common_models(mm1, mm2)
# print(common_models)

# import warnings

from dataset.common import get_separated_sequences_mental_models_dataset
from models.mm_inference import create_inference_model_with_scores

# warnings.filterwarnings("ignore")
import tensorflow as tf
# import keras.backend as kb
import numpy as np
#
# batch_size = 3
# M = tf.random.uniform(shape=[batch_size,4,5])
# S = tf.random.uniform(shape=[batch_size,4])
# y_pred = (M,S)
# y_actual = tf.random.uniform(shape=[batch_size, 2, 5])
#
# alpha = 1
# beta = 1
# M, S = y_pred

# loss_value = kb.sum(
#     alpha * (kb.sum(kb.prod(M, S), axis=1) / kb.max(np.array([1, kb.sum(S, axis=1)])) - y_actual) ** 2 + \
#     beta * kb.max([0, 1 - kb.sum(S, axis=1)]))

# print(M * tf.expand_dims(S, axis=-1))
# print(kb.sum(S, axis=1))
# print(tf.ones(S.shape[0]))
#
# print(tf.stack([tf.ones(S.shape[0]), kb.sum(S, axis=1)], axis=1))
# print(kb.max(tf.stack([tf.ones(S.shape[0]), kb.sum(S, axis=1)], axis=1), axis=1))


# A = kb.sum(M * tf.expand_dims(S, axis=-1), axis=1)
# B = kb.max(tf.stack([tf.ones(S.shape[0]), kb.sum(S, axis=1)], axis=1), axis=1)
# B = kb.maximum(1.0, kb.sum(S, axis=1))


# print(A.shape, B.shape)
# print((A / tf.expand_dims(B, axis=-1) - y_actual))

# MSE = ((A / tf.expand_dims(B, axis=-1)) - y_actual)**2
# scores = 1 - kb.sum(S, axis=1)
#
# loss_value = alpha * MSE + beta * scores

# print(A.shape, B.shape)
# print(A / tf.expand_dims(B, axis=-1))
#
# print()

# print(kb.sum(M * tf.expand_dims(S, axis=-1)) / kb.max(tf.stack([tf.ones(S.shape[0]), kb.sum(S, axis=1)], axis=1), axis=1))

# print(1 - kb.sum(S, axis=1))

import keras as kr
def create_varying_inference_model(num_variables, max_input_length, ds):
    # Without specific token for start of sequence (index 0) and end of sequence (index 1) - (0,0,0,0,0) equals end
    # Based on characters instead of subsentences
    # Initialise parameters
    batch_size = 8
    embedding_size = 10
    hidden_units = 128
    max_sub_mental_models = 3
    mm_l1 = 0.0
    score_l1 = 0.0
    num_operators = 5  # and, or, not
    input_dim = num_variables + num_operators
    max_length = 10
    output_dim = 5

    print('max_input_length', max_input_length)

    encoder_inputs = kr.Input(shape=(2, max_length))
    encoder = create_inference_model_with_scores(hidden_units,
                                       embedding_size,
                                       max_sub_mental_models,
                                       mm_l1,
                                       score_l1,
                                       max_length,
                                       input_dim,
                                       output_dim)

    encoder_mms, encoder_scores = encoder(encoder_inputs)
    nn_flatten = tf.keras.layers.Reshape((max_sub_mental_models**2,output_dim))(encoder_mms)
    nn_concat = tf.keras.layers.Concatenate()([nn_flatten, encoder_scores])
    encoder_states = [nn_concat, nn_concat]

    # Create decoder
    # decoder = kr.layers.LSTM(hidden_units, return_sequences=True, return_state=True, activation='relu')
    # decoder_outputs, _, _ = decoder(nn_concat)
    # decoder_dense = kr.layers.Dense(num_variables, activation='tanh')
    # output = decoder_dense(decoder_outputs)

    #####################################################################################################
    # encoder = kr.layers.LSTM(hidden_units, return_sequences=True, return_state=True, activation='relu')
    # encoder_outputs, state_h, state_c = encoder(nn_flatten)
    # encoder_states = [state_h, state_c]

    # Create decoder
    decoder_inputs = kr.Input(shape=(None,num_variables))
    decoder = kr.layers.LSTM(54, return_sequences=True, return_state=True, activation='relu')
    decoder_outputs, _, _ = decoder(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = kr.layers.Dense(num_variables, activation='tanh')
    output = decoder_dense(decoder_outputs)

#############################################################################################################

    ## Define training model
    model_train = kr.Model(inputs=[encoder_inputs, decoder_inputs], outputs=output)
    model_train.summary()

    ## Train model
    model_train.compile(optimizer=kr.optimizers.Adam(learning_rate=1e-3),
                  loss=kr.losses.mse)

    callbacks = [kr.callbacks.EarlyStopping(patience=20, min_delta=1e-5, restore_best_weights=True)]
    history = model_train.fit([ds.x_train, ds.y_train_d], ds.y_train, validation_data=([ds.x_valid, ds.y_valid_d], ds.y_valid),
                        epochs=1000, batch_size=batch_size, callbacks=callbacks)

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
        temp = np.zeros((data.shape[0],data.shape[1]+1,data.shape[2]))
        temp[:,1:,:] = data
    elif position == 'last':
        temp = np.zeros((data.shape[0],data.shape[1]+1,data.shape[2]))
        temp[:,:-1,:] = data

    return temp

import dataset
import matplotlib.pyplot as plt
if __name__ == '__main__':
    ds = get_separated_sequences_mental_models_dataset('../data', 'encoded_and_trees_single_mms_type_I',
                                                       num_variables=5, max_depth=2,
                                                       test_size=.1, valid_size=.1,
                                                       indexed_encoding=True, pad_mental_models=True)

    dec_in, dec_out = dataset.encoding.create_decoding_dictionaries(ds.input_dictionary, ds.output_dictionary)

    ds.y_train_d = add_zero_row(ds.y_train, 'front')
    ds.y_train = add_zero_row(ds.y_train, 'last')
    ds.y_valid_d = add_zero_row(ds.y_valid, 'front')
    ds.y_valid = add_zero_row(ds.y_valid, 'last')
    ds.y_test_d = add_zero_row(ds.y_test, 'front')
    ds.y_test = add_zero_row(ds.y_test, 'last')

    num_variables = 5
    num_operators = 5  # and, or, not
    num_symbols = num_variables + num_operators
    max_input_length = ds.x_train.shape[-1]

    # 1: subsentences no index, 2: subsentences index, 3: symbols no index
    model_train, history, encoder_model, decoder_model = create_varying_inference_model(num_variables,
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
