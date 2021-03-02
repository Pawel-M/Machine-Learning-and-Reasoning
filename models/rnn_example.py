import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import dataset.encoding


data = dataset.encoding.load_sentences_and_conclusions('../data', 2, 5)
sentences, conclusions, input_dictionary, output_dictionary = data
encoded_sentences = [np.where(sentence == 1)[1] for sentence in sentences]

x = kr.preprocessing.sequence.pad_sequences(encoded_sentences, padding='post')
y = np.array(conclusions)

input_dim = x.shape[-1]
output_dim = y.shape[-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=1337)

model = kr.Sequential()
model.add(kr.layers.Embedding(input_dim=input_dim, mask_zero=True, output_dim=64))
model.add(kr.layers.LSTM(units=128))
model.add(kr.layers.Dense(output_dim, activation='softmax'))
model.summary()

learning_rate = 0.001
batch_size = 64
epochs = 50

model.compile(optimizer=kr.optimizers.Adam(learning_rate=learning_rate),
              loss=kr.losses.categorical_crossentropy,
              metrics=['categorical_accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

pred = model.predict(x_test)
test_accuracy = np.mean(kr.metrics.categorical_accuracy(y_test, pred))
print(f'Test accuracy: {test_accuracy * 100:.1f} %')

loss = history.history['loss']
accuracy = history.history['categorical_accuracy']

fig, ax = plt.subplots()
ax.plot(range(len(loss)), loss, label='loss', color='tab:blue')
ax.set_ylabel('loss', color='tab:blue')

ax2 = ax.twinx()
ax2.plot(range(len(accuracy)), accuracy, color='tab:orange')
ax2.set_ylabel('accuracy', color='tab:orange')

ax.set_xlabel('epoch')
plt.show()
