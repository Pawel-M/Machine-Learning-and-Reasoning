"""
From: https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
"""
import tensorflow as tf
import tensorflow.keras as kr


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask zero out padding tokens.
    if mask is not None:
        logits += (mask * -1e9)

    attention_scores = tf.nn.softmax(logits, axis=-1)

    return tf.matmul(attention_scores, value), attention_scores


class MultiHeadAttention(kr.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = kr.layers.Dense(units=d_model)
        self.key_dense = kr.layers.Dense(units=d_model)
        self.value_dense = kr.layers.Dense(units=d_model)

        self.dense = kr.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, attention_scores = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)

        return outputs, attention_scores

    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'name': self.name,
        }


class PositionalEncoding(kr.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        return {
            'position': self.position,
            'd_model': self.d_model,
        }


# def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
#     inputs = kr.Input(shape=(None, d_model), name="inputs")
#     enc_outputs = kr.Input(shape=(None, d_model), name="encoder_outputs")
#     look_ahead_mask = kr.Input(
#         shape=(1, None, None), name="look_ahead_mask")
#     padding_mask = kr.Input(shape=(1, 1, None), name='padding_mask')
#
#     attention1 = MultiHeadAttention(
#         d_model, num_heads, name="attention_1")(inputs={
#         'query': inputs,
#         'key': inputs,
#         'value': inputs,
#         'mask': look_ahead_mask
#     })
#     attention1 = kr.layers.LayerNormalization(
#         epsilon=1e-6)(attention1 + inputs)
#
#     attention2 = MultiHeadAttention(
#         d_model, num_heads, name="attention_2")(inputs={
#         'query': attention1,
#         'key': enc_outputs,
#         'value': enc_outputs,
#         'mask': padding_mask
#     })
#     attention2 = kr.layers.Dropout(rate=dropout)(attention2)
#     attention2 = kr.layers.LayerNormalization(
#         epsilon=1e-6)(attention2 + attention1)
#
#     outputs = kr.layers.Dense(units=units, activation='relu')(attention2)
#     outputs = kr.layers.Dense(units=d_model)(outputs)
#     outputs = kr.layers.Dropout(rate=dropout)(outputs)
#     outputs = kr.layers.LayerNormalization(
#         epsilon=1e-6)(outputs + attention2)
#
#     return kr.Model(
#         inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
#         outputs=outputs,
#         name=name)
#
#
# def decoder(vocab_size,
#             num_layers,
#             units,
#             d_model,
#             num_heads,
#             dropout,
#             name='decoder'):
#     inputs = kr.Input(shape=(None,), name='inputs')
#     enc_outputs = kr.Input(shape=(None, d_model), name='encoder_outputs')
#     look_ahead_mask = kr.Input(
#         shape=(1, None, None), name='look_ahead_mask')
#     padding_mask = kr.Input(shape=(1, 1, None), name='padding_mask')
#
#     embeddings = kr.layers.Embedding(vocab_size, d_model)(inputs)
#     embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
#     embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
#
#     outputs = kr.layers.Dropout(rate=dropout)(embeddings)
#
#     for i in range(num_layers):
#         outputs = decoder_layer(
#             units=units,
#             d_model=d_model,
#             num_heads=num_heads,
#             dropout=dropout,
#             name='decoder_layer_{}'.format(i),
#         )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])
#
#     return kr.Model(
#         inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
#         outputs=outputs,
#         name=name)
#
#
# def transformer(vocab_size,
#                 num_layers,
#                 units,
#                 d_model,
#                 num_heads,
#                 dropout,
#                 name="transformer"):
#     inputs = kr.Input(shape=(None,), name="inputs")
#     dec_inputs = kr.Input(shape=(None,), name="dec_inputs")
#
#     enc_padding_mask = kr.layers.Lambda(
#         create_padding_mask, output_shape=(1, 1, None),
#         name='enc_padding_mask')(inputs)
#     # mask the future tokens for decoder inputs at the 1st attention block
#     look_ahead_mask = kr.layers.Lambda(
#         create_look_ahead_mask,
#         output_shape=(1, None, None),
#         name='look_ahead_mask')(dec_inputs)
#     # mask the encoder outputs for the 2nd attention block
#     dec_padding_mask = kr.layers.Lambda(
#         create_padding_mask, output_shape=(1, 1, None),
#         name='dec_padding_mask')(inputs)
#
#     enc_outputs = encoder(
#         vocab_size=vocab_size,
#         num_layers=num_layers,
#         units=units,
#         d_model=d_model,
#         num_heads=num_heads,
#         dropout=dropout,
#     )(inputs=[inputs, enc_padding_mask])
#
#     dec_outputs = decoder(
#         vocab_size=vocab_size,
#         num_layers=num_layers,
#         units=units,
#         d_model=d_model,
#         num_heads=num_heads,
#         dropout=dropout,
#     )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
#
#     outputs = kr.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)
#
#     return kr.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


def encoder_layer(inputs, padding_mask, units, d_model, num_heads, dropout, name="encoder_layer"):
    # inputs = kr.Input(shape=(None, d_model), name="inputs")
    # padding_mask = kr.Input(shape=(1, 1, None), name="padding_mask")

    attention, attention_scores = MultiHeadAttention(
        d_model, num_heads, name="attention")({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })
    attention = kr.layers.Dropout(rate=dropout)(attention)
    attention = kr.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = kr.layers.Dense(units=units, activation='relu')(attention)
    outputs = kr.layers.Dense(units=d_model)(outputs)
    outputs = kr.layers.Dropout(rate=dropout)(outputs)
    outputs = kr.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)

    # return kr.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
    return outputs, attention_scores


def encoder(inputs,
            padding_mask,
            max_length,
            vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    # inputs = kr.Input(shape=(max_length,), name="inputs")
    # padding_mask = kr.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = kr.layers.Embedding(vocab_size + 1, d_model, mask_zero=True)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(max_length, d_model)(embeddings)

    outputs = kr.layers.Dropout(rate=dropout)(embeddings)
    all_attention_scores = []
    for i in range(num_layers):
        outputs, attention_scores = encoder_layer(
            inputs=outputs,
            padding_mask=padding_mask,
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )
        all_attention_scores.append(attention_scores)

    # return kr.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
    return outputs, all_attention_scores


def transformer_encoder_only(max_length,
                             input_dim,
                             output_dim,
                             num_layers,
                             units,
                             d_model,
                             num_heads,
                             dropout,
                             name='transformer_encoder_only'):
    inputs = kr.Input(shape=(max_length,), name='inputs')
    enc_padding_mask = kr.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)

    enc_outputs, all_attention_scores = encoder(
        inputs=inputs,
        padding_mask=enc_padding_mask,
        max_length=max_length,
        vocab_size=input_dim,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )
    # concat_attention = tf.reshape(scaled_attention,
    #                               (batch_size, -1, self.d_model))
    flatten_outputs = kr.layers.Flatten()(enc_outputs)
    outputs = kr.layers.Dense(units=output_dim, activation='softmax', name="outputs")(flatten_outputs)

    base_model = kr.Model(inputs=inputs, outputs=[outputs, all_attention_scores], name=name)
    train_model = kr.Model(inputs=inputs, outputs=outputs, name=name)
    return train_model, base_model


CUSTOM_OBJECTS = {
    'PositionalEncoding': PositionalEncoding,
    'MultiHeadAttention': MultiHeadAttention,
}

if __name__ == '__main__':
    import numpy as np
    from dataset.common import get_dataset

    dataset = get_dataset('../data', depth=2, num_variables=5, test_size=.1, valid_size=.1, indexed_encoding=True)
    max_length = dataset.x_train.shape[-1]
    input_dim = np.max(dataset.x_train)
    output_dim = dataset.y_train.shape[-1]
    # input_dim = 10
    # output_dim = 5
    print('max_length', max_length)
    print('input_dim', input_dim)
    print('output_dim', output_dim)

    num_layers = 1
    units = 64
    d_model = 64
    num_heads = 8
    dropout = 0.01

    learning_rate = 0.001
    batch_size = 64
    epochs = 3
    patience = 10
    min_delta = 1e-4

    train_model, base_model = transformer_encoder_only(max_length=max_length,
                                                       input_dim=input_dim, output_dim=output_dim,
                                                       num_layers=num_layers, units=units,
                                                       d_model=d_model, num_heads=num_heads, dropout=dropout)
    base_model.summary()

    train_model.compile(optimizer=kr.optimizers.Adam(learning_rate),
                        loss=kr.losses.categorical_crossentropy,
                        metrics=['categorical_accuracy']
                        )
    train_model.summary()

    callbacks = [kr.callbacks.EarlyStopping(patience=patience,
                                            min_delta=min_delta,
                                            restore_best_weights=True,
                                            verbose=1)]
    history = train_model.fit(dataset.x_train, dataset.y_train, validation_data=(dataset.x_valid, dataset.y_valid),
                              batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    preds = base_model.predict(dataset.x_test)
    attention_scores = preds[1]
    print(attention_scores[0].shape)
