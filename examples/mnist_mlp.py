'''Train a simple deep NN on the MNIST dataset.

Get to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.utils import np_utils

nb_classes = 10
# the data, shuffled and split between tran and test sets
(D_X_train, D_y_train), (D_X_test, D_y_test) = mnist.load_data()

D_X_train = D_X_train.reshape(60000, 784)
D_X_test = D_X_test.reshape(10000, 784)
D_X_train = D_X_train.astype('float32')
D_X_test = D_X_test.astype('float32')
D_X_train /= 255
D_X_test /= 255
print(D_X_train.shape[0], 'train samples')
print(D_X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
D_Y_train = np_utils.to_categorical(D_y_train, nb_classes)
D_Y_test = np_utils.to_categorical(D_y_test, nb_classes)


import dlx.unit.core as U
from dlx.model import Model

input_size = 784
hidden_size = 512
output_size = nb_classes
batch_size = 128
nb_epoch = 20

'''Define Units'''
#Data unit
data = U.Input(2, 'X')
#Dense unit 1
dense_1 = U.Dense(input_size, hidden_size, 'Dense1')
#Activation unit 1
activation_1 = U.Activation('relu')
#Dense unit 2
dense_2 = U.Dense(hidden_size, hidden_size, 'Dense2')
#Activation unit 2
activation_2 = U.Activation('relu')
#Dense unit 3
dense_3 = U.Dense(hidden_size, output_size, 'Dense3')
#Activation unit 3
activation_3 = U.Activation('softmax')
#Output unit
output = U.Output()

'''Define Relations'''
dense_1.set_input('input', data, 'output')
activation_1.set_input('input', dense_1, 'output')
dense_2.set_input('input', activation_1, 'output')
activation_2.set_input('input', dense_2, 'output')
dense_3.set_input('input', activation_2, 'output')
activation_3.set_input('input', dense_3, 'output')
output.set_input('input', activation_3, 'output')

'''Build Model'''
model = Model()
model.add_input(data, 'X')
model.add_output(output, 'y')
model.add_hidden(dense_1)
model.add_hidden(dense_2)
model.add_hidden(dense_3)
model.add_hidden(activation_1)
model.add_hidden(activation_2)
model.add_hidden(activation_3)


model.compile(optimizer='rmsprop', loss_configures = [('y', 'categorical_crossentropy', None, False, "categorical"),], verbose=0)
model.fit(data = {'X': D_X_train, 
                  'y': D_Y_train},
          batch_size=batch_size, 
          nb_epoch=nb_epoch,
          show_accuracy=True, 
          verbose=2,
          validation_data={'X': D_X_test, 
                           'y': D_Y_test}
          )
score = model.evaluate(data = {'X': D_X_test, 
                           'y': D_Y_test},
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

