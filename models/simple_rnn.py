from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow.keras as kr
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

import dataset.encoding

def next_batch(x, y, seq_len, batch_size):
    N = x.shape[0]
    batch_indeces = np.random.permutation(N)[:batch_size]
    x_batch = x[batch_indeces]
    y_batch = y[batch_indeces]
    seq_len_batch = seq_len[batch_indeces]
    return x_batch, y_batch, seq_len_batch

def add_padding(sentences):
    for i, x in enumerate(sentences):
        if x.shape[0] < seq_max_len:
            add_num_rows = seq_max_len - x.shape[0]
            sentences[i] = np.concatenate((x, np.zeros((add_num_rows, input_dim))))

    return(sentences)

def get_best_index(outputs):
    for i, out in enumerate(outputs):
        max_ind = np.argmax(out)
        result = np.zeros(len(out))
        result[max_ind] = 1
        outputs[i] = result
    return outputs

def calc_acc(true,pred):
    return sum([np.all(pred[i] == true[i]) for i in range(len(true))]) / len(true) * 100


data = dataset.encoding.load_sentences_and_conclusions('../data', 2, 5)
sentences, conclusions, input_dictionary, output_dictionary = data

seq_max_len = max([x.shape[0] for x in sentences])
input_dim = sentences[0].shape[1]
out_dim = len(conclusions[0])

x = sentences
y = np.array(conclusions)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=1337)

seq_len_train = np.array([x.shape[0] for x in x_train])
seq_len_test = np.array([x.shape[0] for x in x_test])

x_train = np.stack(add_padding(x_train))
x_test = np.stack(add_padding(x_test))

## 2
# Parameters
learning_rate = 0.01    # The optimization initial learning rate
training_steps = 10000  # Total number of training steps
batch_size = 10         # batch size
display_freq = 1000     # Frequency of displaying the training results
num_hidden_units = 20   # number of hidden units

## 3 Model
# weight and bais wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initial)

def RNN(x, weights, biases, n_hidden, seq_max_len, seq_len):
    """
    :param x: inputs of shape [batch_size, max_time, input_dim]
    :param weights: matrix of fully-connected output layer weights
    :param biases: vector of fully-connected output layer biases
    :param n_hidden: number of hidden units
    :param seq_max_len: sequence maximum length
    :param seq_len: length of each sequence of shape [batch_size,]
    """
    cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_len, dtype=tf.float32)

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seq_len - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    out = tf.matmul(outputs, weights) + biases
    return out

## 4
# Placeholders for inputs(x), input sequence lengths (seqLen) and outputs(y)
x = tf.placeholder(tf.float32, [None, seq_max_len, input_dim])
seqLen = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.float32, [None, out_dim])

# create weight matrix initialized randomly from N~(0, 0.01)
W = weight_variable(shape=[num_hidden_units, out_dim])
# create bias vector initialized as zero
b = bias_variable(shape=[out_dim])

# Network predictions
pred_out = RNN(x, W, b, num_hidden_units, seq_max_len, seqLen)

# Define the loss function (i.e. mean-squared error loss) and optimizer
cost = tf.reduce_mean(tf.square(pred_out - y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

## 5
with tf.Session() as sess:
    sess.run(init)
    print('----------Training---------')
    for i in range(training_steps):
        x_batch, y_batch, seq_len_batch = next_batch(x_train, y_train, seq_len_train, batch_size)
        _, mse = sess.run([train_op, cost], feed_dict={x: x_batch, y: y_batch, seqLen: seq_len_batch})
        if (i + 1) % display_freq == 0:
            print('Step {0:<6}, MSE={1:.4f}'.format(i+1, mse))

    # Test
    y_pred = sess.run(pred_out, feed_dict={x: x_test, seqLen: seq_len_test})
    y_pred = get_best_index(y_pred)
    print('--------Test Results-------')
    for i, x in enumerate(y_test):
        print("When the ground truth output is {}, the model thinks it is {}"
              .format(y_test[i], y_pred[i]))

    print('\n--------Accuracy-------')
    print('The accuracy of the current model is {:.1f}%'.format(calc_acc(y_test,y_pred)))



