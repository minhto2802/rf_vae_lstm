import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras_vae_example.tsc.vae_model import create_lstm_vae
from keras.utils.vis_utils import plot_model
from keras import optimizers
import pylab as plt

import numpy as np

to_plot_model = False

batch_size = 100
intermediate_dim = 256
latent_dim = 100
epochs = 80
lr = 1e-3

# x = np.random.normal(0, 1, (1000, 100, 1))


def get_data():
    # read data from file
    data = np.fromfile('sample_data.dat').reshape(419,13)
    timesteps = 3

    dataX = []
    for i in range(len(data) - timesteps - 1):
        x = data[i:(i+timesteps), :]
        dataX.append(x)
    return np.array(dataX)


x = get_data()
print('Input shape: ', x.shape)

input_dim = x.shape[-1]
timesteps = x.shape[1]
# steps_per_epoch = x.shape[0] // batch_size

model = create_lstm_vae(input_dim,
                    timesteps,
                    batch_size,
                    intermediate_dim,
                    latent_dim,
                    epsilon_std=1.)

if to_plot_model:
    plot_model(model[0], to_file='vae_lstm_model.png', show_shapes=True)

# print('\nEncoder:')
# model[1].summary()
# z = model[1].predict(x)
# print('Encoder output shape: ', z.shape)
#
# print('\nDecoder:')
# model[2].summary()
# print('Decoder output shape: ', model[2].predict(z).shape)

print('\nWhole model: ')
model[0].fit(x, x, epochs=epochs, batch_size=batch_size)
model[0].summary()
preds = model[0].predict(x, batch_size=batch_size)
print('Final output shape: ', preds.shape)

# plt.plot(x[0, :, 0], label='data')
# plt.plot(preds[0, :, 0], label='predict')
plt.plot(x[:, 0, 3], label='data')
plt.plot(preds[:, 0, 3], label='predict')

plt.legend()
plt.show()
