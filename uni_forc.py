from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['figure.dpi'] = 500
mpl.rcParams['axes.grid'] = False
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }

df = pd.read_csv('load_data.csv')


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


TRAIN_SPLIT = 95000
tf.random.set_seed(13)

uni_data = df['ERCOT']
uni_data.index = df['Hour_End']

uni_data = uni_data.values

uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data-uni_train_mean)/uni_train_std
univariate_past_history = 24
univariate_future_target = 1

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)

x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['Historical', 'True Value', 'Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title, font2)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend(prop=font2, loc=1)
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.xlabel('Time-Step', font1)
    return plt


BATCH_SIZE = 256
BUFFER_SIZE = 500

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(20, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)])

simple_lstm_model.compile(optimizer='adam', loss='msle')

EVALUATION_INTERVAL = 200
EPOCHS = 60

history = simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)

plt.plot(history.history['loss'], '-',linewidth=2.0,label='Training loss')
plt.plot(history.history['val_loss'], '-.',linewidth=2.0,label='Validation loss')

plt.ylabel('Loss', font1)
plt.xlabel('Epoch', font1)
plt.legend(prop=font2, loc=(6.5/14, 0.3/0.5))
plt.xticks(fontproperties='Times New Roman', size=20)
plt.yticks(fontproperties='Times New Roman', size=20)

plt.show()

for x, y in val_univariate.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
    simple_lstm_model.predict(x)[0]], 0, 'One hour ahead prediction')
    plot.show()


