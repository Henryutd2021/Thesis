from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['axes.grid'] = False
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }


df = pd.read_csv('load_data_with weather without nan.csv')


TRAIN_SPLIT = 100000
tf.random.set_seed(13)
BATCH_SIZE = 256
BUFFER_SIZE = 500
EVALUATION_INTERVAL = 200
EPOCHS = 3
past_history = 24
future_target = 6
STEP = 1


#features_considered = ['ERCOT']
features_considered = ['COAST', 'ERCOT', 'Temperature', 'Dewpoint', 'RH', 'Windspeed']
features = df[features_considered]
features.index = df['Hour_End']


dataset = features.values


data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)

data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std
#print(dataset)

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=True):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)


x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP,single_step=False)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP,single_step=False)

# print('Single window of past history : {}'.format(x_train_multi[0].shape))
# print('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))


train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()



def create_time_steps(length):
    return list(range(-length, 0))


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, '-',linewidth=2.0,label='training loss')
    plt.plot(epochs, val_loss, '--',linewidth=2.0,label='validation loss')
    plt.ylabel('Loss', font1)
    plt.xlabel('Epoch', font1)
    plt.legend(prop=font2)
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.title(title, font2)
    plt.show()


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(10, 5))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), '--', label='Historical')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'b-',
           label='True Value')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'r-.',
            label='Prediction')
    #plt.legend(loc='upper left')
    plt.legend(prop=font2, loc='best')
    #plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.xlabel('Time-Step', font1)
    plt.title('Six hour ahead prediction', font2)
    plt.show()


multi_step_model = tf.keras.models.Sequential()
#multi_step_model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='SAME',input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.Dropout(0.2))

multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
#multi_step_model.add(tf.keras.layers.Dropout(0.2))
multi_step_model.add(tf.keras.layers.Dense(6))



multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')


multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)


plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

for x, y in val_data_multi.take(3):
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])


