import numpy as np
import tensorflow as tf
import tensorflow.keras as kr


def broadcast(x, y):
    tf.print(x.shape, y.shape)
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]
    x = np.transpose(x, axes=[0, 2, 1])
    y = np.transpose(y, axes=[2, 0, 1])
    x, y = np.broadcast_arrays(x, y)
    return x, y


def calculate_values(x, y):
    s = x + y
    sc = np.clip(s, -1, 1)
    return sc


def calculate_correctness(x, y):
    diff = 1 - np.maximum(0, np.abs(x - y) - 1)
    prod = np.prod(diff, axis=-1)
    return prod


def calculate_values_soft(x, y, av=10):
    return np.tanh((x + y) * av)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_correctness_soft(x, y, ac=10):
    diff = 1 - sigmoid((np.abs(x - y) - 1.5) * ac)
    prod = np.prod(diff, axis=-1)
    return prod


def calculate_out(values, correctness):
    result = values * correctness[..., np.newaxis]
    reshaped = np.reshape(result, (result.shape[0] * result.shape[1], result.shape[2]))
    return reshaped


def combine_mental_models(mm1, mm2):
    mm1b, mm2b = broadcast(mm1, mm2)
    values = calculate_values(mm1b, mm2b)
    correctness = calculate_correctness(mm1b, mm2b)
    out = calculate_out(values, correctness)
    return out


def combine_mental_models_soft(mm1, mm2):
    mm1b, mm2b = broadcast(mm1, mm2)
    values = calculate_values_soft(mm1b, mm2b, av=10)
    correctness = calculate_correctness_soft(mm1b, mm2b, ac=10)
    out = calculate_out(values, correctness)
    return out


def test_mm_inference():
    # (a or b)      ---> [T, n], [n, T]
    # (a or not b)  ---> [T, n], [n, F]
    mm1 = np.array([
        [1, 0],
        [0, 1],
    ])
    mm2 = np.array([
        [1, 0],
        [0, -1]
    ])

    combined_mental_models = combine_mental_models(mm1, mm2)
    combined_mental_models_soft = combine_mental_models_soft(mm1, mm2)
    print(combined_mental_models)
    print(combined_mental_models_soft)


class MMInferenceLayer(kr.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def broadcast(self, x, y):
        x = tf.expand_dims(x, axis=-1)
        y = tf.expand_dims(y, axis=-1)
        x = tf.transpose(x, perm=[0, 1, 3, 2])
        y = tf.transpose(y, perm=[0, 3, 1, 2])
        # x = tf.broadcast_to(x, (x.shape[0], x.shape[1]))
        # x, y =  np.broadcast_arrays(x, y)
        return x, y

    def call(self, inputs, **kwargs):
        x = inputs[0]
        y = inputs[1]
        x, y = self.broadcast(x, y)

        # calculate value
        s = x + y
        value = tf.clip_by_value(s, -1, 1)

        # calculate correctness
        diff = 1 - tf.maximum(0., tf.abs(x - y) - 1.)
        correctness = tf.reduce_prod(diff, axis=-1)

        # calculate mental models
        mms = value * tf.expand_dims(correctness, axis=-1)

        # sum and normalize to obtain a single mental model
        reshaped_value = tf.reshape(mms, (-1, mms.shape[-3] * mms.shape[-2], mms.shape[-1]))
        reshaped_correctness = tf.reshape(correctness, (-1, correctness.shape[-2] * correctness.shape[-1]))
        mm = tf.reduce_sum(reshaped_value, axis=-2)
        # normalization by correctness
        mm = mm / tf.reduce_sum(reshaped_correctness, axis=-1, keepdims=True)

        # mm = tf.clip_by_value(mm, -1, 1)
        # mm = tf.tanh(mm)
        return mm


class MMInferenceScoresLayer(kr.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def broadcast(self, x, y):
        x = tf.expand_dims(x, axis=-1)
        y = tf.expand_dims(y, axis=-1)
        x = tf.transpose(x, perm=[0, 1, 3, 2])
        y = tf.transpose(y, perm=[0, 3, 1, 2])
        return x, y

    def broadcast_scores(self, x_scores, y_scores):
        x_scores = tf.expand_dims(x_scores, axis=-1)
        y_scores = tf.expand_dims(y_scores, axis=-1)
        y_scores = tf.transpose(y_scores, perm=[0, 2, 1])
        return x_scores, y_scores

    def call(self, inputs, **kwargs):
        x, x_scores = inputs[0]
        y, y_scores = inputs[1]

        x, y = self.broadcast(x, y)
        x_scores, y_scores = self.broadcast_scores(x_scores, y_scores)

        # calculate value
        s = x + y
        value = tf.clip_by_value(s, -1, 1)

        # calculate correctness
        diff = 1 - tf.maximum(0., tf.abs(x - y) - 1.)
        correctness = tf.reduce_prod(diff, axis=-1)

        # calculate mental models
        mms = value * tf.expand_dims(correctness, axis=-1)

        # calculate and apply scores
        scores = x_scores * y_scores
        mms = mms * tf.expand_dims(scores, axis=-1)

        # sum and normalize to obtain a single mental model
        reshaped_value = tf.reshape(mms, (-1, mms.shape[-3] * mms.shape[-2], mms.shape[-1]))
        mm = tf.reduce_sum(reshaped_value, axis=-2)

        # normalization by correctness
        # reshaped_correctness = tf.reshape(correctness, (-1, correctness.shape[-2] * correctness.shape[-1]))
        reshaped_correctness = tf.reshape(correctness * scores, (-1, correctness.shape[-2] * correctness.shape[-1]))
        mm = mm / tf.reduce_sum(reshaped_correctness, axis=-1, keepdims=True)

        # mm = tf.clip_by_value(mm, -1, 1)
        # mm = tf.tanh(mm)
        return mm


class MultiMMInferenceLayer(kr.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def broadcast(self, x, y):
        x = tf.expand_dims(x, axis=-1)
        y = tf.expand_dims(y, axis=-1)
        x = tf.transpose(x, perm=[0, 1, 3, 2])
        y = tf.transpose(y, perm=[0, 3, 1, 2])
        return x, y

    def broadcast_scores(self, x_scores, y_scores):
        x_scores = tf.expand_dims(x_scores, axis=-1)
        y_scores = tf.expand_dims(y_scores, axis=-1)
        y_scores = tf.transpose(y_scores, perm=[0, 2, 1])
        return x_scores, y_scores

    def call(self, inputs, **kwargs):
        x, x_scores = inputs[0]
        y, y_scores = inputs[1]

        x, y = self.broadcast(x, y)
        x_scores, y_scores = self.broadcast_scores(x_scores, y_scores)

        # calculate value
        s = x + y
        value = tf.clip_by_value(s, -1, 1)

        # calculate correctness
        diff = 1 - tf.maximum(0., tf.abs(x - y) - 1.)
        correctness = tf.reduce_prod(diff, axis=-1)

        # calculate mental models
        mms = value
        # mms = value * tf.expand_dims(correctness, axis=-1)

        # calculate and apply scores
        scores = x_scores * y_scores
        # mms = mms * tf.expand_dims(scores, axis=-1)

        mms = tf.reshape(mms, (-1, mms.shape[-3] * mms.shape[-2], mms.shape[-1]))
        correctness = tf.reshape(correctness, (-1, correctness.shape[-2] * correctness.shape[-1]))
        scores = tf.reshape(scores, (-1, scores.shape[-2] * scores.shape[-1]))

        return mms, correctness * scores


def create_inference_model(hidden_units,
                           embedding_size,
                           max_sub_mental_models,
                           mm_l1,
                           max_length,
                           input_dim,
                           output_dim):
    print('max_input_length', max_length)
    input = kr.Input(shape=(2, max_length))
    split_layer = kr.layers.Lambda(lambda x: (x[:, 0], x[:, 1]))(input)

    nn_input = kr.Input(max_length)
    nn_embedding_layer = kr.layers.Embedding(input_dim + 1, embedding_size)(nn_input)
    flatten_layer = kr.layers.Flatten()(nn_embedding_layer)
    nn_hidden = kr.layers.Dense(hidden_units, activation='relu')(flatten_layer)
    nn_output = kr.layers.Dense(output_dim * max_sub_mental_models,
                                activation='tanh',
                                activity_regularizer=kr.regularizers.L1(mm_l1))(nn_hidden)
    nn_reshape = kr.layers.Reshape((max_sub_mental_models, output_dim))(nn_output)
    sub_sequence_nn = kr.Model(inputs=nn_input, outputs=nn_reshape, name='sub-sequence-NN')
    sub_sequence_nn.summary()

    mm = sub_sequence_nn(split_layer[0]), sub_sequence_nn(split_layer[1])
    mm_inference_layer = MMInferenceLayer()(mm)

    model = kr.Model(inputs=input, outputs=mm_inference_layer)
    model.summary()

    return model


def create_inference_model_with_scores(hidden_units,
                                       embedding_size,
                                       max_sub_mental_models,
                                       mm_l1,
                                       score_l1,
                                       max_length,
                                       input_dim,
                                       output_dim):
    print('max_input_length', max_length)
    input = kr.Input(shape=(2, max_length))
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
    mm_inference_layer = MMInferenceScoresLayer()(mms_and_scores)

    model = kr.Model(inputs=input, outputs=mm_inference_layer)
    model.summary()

    return model


def create_multi_mm_inference_model(hidden_units,
                                    embedding_size,
                                    max_sub_mental_models,
                                    mm_l1,
                                    score_l1,
                                    max_length,
                                    input_dim,
                                    output_dim):
    print('max_input_length', max_length)
    input = kr.Input(shape=(2, max_length))
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
            if type(pred) is list:
                print(np.rint(pred[0]), np.rint(pred[1]))
            else:
                print(np.rint(pred))
            print(pred)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from models.training import mental_model_accuracy

    # xs, ys = create_ds()

    from dataset.common import get_separated_sequences_mental_models_dataset
    import dataset.encoding

    ds = get_separated_sequences_mental_models_dataset('./data', 'encoded_and_trees_single_mms',
                                                       num_variables=5, max_depth=2,
                                                       test_size=.1, valid_size=.1,
                                                       indexed_encoding=True, pad_mental_models=True)

    dec_in, dec_out = dataset.encoding.create_decoding_dictionaries(ds.input_dictionary, ds.output_dictionary)

    # ds.y_train = ds.y_train[..., 0, :]
    # ds.y_valid = ds.y_valid[..., 0, :]
    # ds.y_test = ds.y_test[..., 0, :]

    num_variables = 5
    num_operators = 5  # and, or, not
    input_dim = num_variables + num_operators
    max_input_length = ds.x_train.shape[-1]

    hidden_units = 1024
    embedding_size = 64
    max_sub_mental_models = 3
    mm_l1 = 0.0
    score_l1 = 0.0

    batch_size = 64

    model = create_multi_mm_inference_model(hidden_units,
                                            embedding_size,
                                            max_sub_mental_models,
                                            mm_l1,
                                            score_l1,
                                            max_length=max_input_length,
                                            input_dim=input_dim,
                                            output_dim=num_variables)

    model.compile(optimizer=kr.optimizers.Adam(learning_rate=1e-3),
                  loss=kr.losses.mse, metrics=[mental_model_accuracy])

    # callbacks = [kr.callbacks.EarlyStopping(patience=20, min_delta=1e-5, restore_best_weights=True)]
    # history = model.fit(ds.x_train, ds.y_train, validation_data=(ds.x_valid, ds.y_valid),
    #                     epochs=1000, batch_size=batch_size, callbacks=callbacks)
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # plt.plot(range(len(loss)), loss, label='loss')
    # plt.plot(range(len(val_loss)), loss, label='val_loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()

    preds = model.predict(ds.x_test)
    preds_int = np.rint(preds[0])
    for i in range(preds[0].shape[0]):
        print(preds_int[i], preds[0][i], preds[1][i], ds.y_test[i])

    # print('errors:')
    # for i in range(preds_int.shape[0]):
    #     if np.sum(np.abs(preds_int[i] - ds.y_test[i])) == 0:
    #         continue
    #     print(dataset.encoding.decode_sentence(ds.x_test[i][0], dec_in, ds.indexed_encoding))
    #     print(dataset.encoding.decode_sentence(ds.x_test[i][1], dec_in, ds.indexed_encoding))
    #     print(preds_int[i], ds.y_test[i])
    #     print(preds[i])
    #     show_subsentence_inference(model, ds, dec_in, [i])
    #     print()

    # errors = np.count_nonzero(np.sum(np.abs(preds_int - ds.y_test), axis=-1))
    # print('errors', int(errors))
    # print(f'accuracy: {mental_model_accuracy(ds.y_test, preds) * 100:.2f}%')
