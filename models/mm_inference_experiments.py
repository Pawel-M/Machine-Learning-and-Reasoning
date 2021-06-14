from dataset.common import get_separated_sequences_mental_models_dataset
from models.mm_inference import MultiMMInferenceLayer

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

    model = kr.Model(inputs=input, outputs=mm_inference_layer)
    model.summary()

    return model


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
    encoder = create_multi_mm_inference_model(encoder_inputs, hidden_units,
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
    nn_flatten = tf.keras.layers.Reshape((encoder_mms.shape[-2]*encoder_mms.shape[-1],))(encoder_mms)
    nn_concat = tf.keras.layers.Concatenate(axis=1)([nn_flatten, encoder_scores])
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
    ds = get_separated_sequences_mental_models_dataset('../data', 'encoded_and_trees_multiple_mms',
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
