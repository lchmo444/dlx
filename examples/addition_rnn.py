# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from data.number_data_engine import NumberDataEngine
from data.character_data_engine import CharacterDataEngine
from dlx.model import slice_X
np.random.seed(1337)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset
TRAINING_SIZE = 50000
DIGITS = 3
INVERT = True
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
MAXLEN = DIGITS + 1 + DIGITS

print('Generating data...')
engine = NumberDataEngine(min_digits=1, max_digits=DIGITS)
questions, expected = engine.get_dataset(TRAINING_SIZE)
print('Total addition questions:', len(questions))

print('Vectorization...')
convertor = CharacterDataEngine(engine.get_character_set(), maxlen=MAXLEN)
D_X = convertor.encode_dataset(questions, invert=True)
D_y = convertor.encode_dataset(expected, maxlen=DIGITS + 1)

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
indices = np.arange(len(D_y))
np.random.shuffle(indices)
D_X = D_X[indices]
D_y = D_y[indices]

# Explicitly set apart 10% for validation data that we never train over
split_at = len(D_X) - len(D_X) / 10
(D_X_train, D_X_val) = (slice_X(D_X, 0, split_at), slice_X(D_X, split_at))
(D_y_train, D_y_val) = (D_y[:split_at], D_y[split_at:])

print(D_X_train.shape)
print(D_y_train.shape)


import dlx.unit.core as U
import dlx.unit.recurrent as R
from dlx.model import Model

print('Build model...')
input_dim = convertor.get_dim()
output_dim = convertor.get_dim()
hidden_dim = HIDDEN_SIZE
input_length = MAXLEN
output_length = DIGITS + 1

'''Define Units'''
#Data uayer
data = U.Input(3, 'X')
#RNN encoder
encoder = R.RNN(input_length, input_dim, hidden_dim, name='ENCODER')
#encoder = R.LSTM(input_length, input_dim, hidden_dim, name='ENCODER')
#RNN decoder
decoder = R.RNN(output_length, hidden_dim, hidden_dim, name='DECODER')
#decoder = R.LSTM(output_length, hidden_dim, hidden_dim, name='DECODER')
#Time Distributed Dense
tdd = U.TimeDistributedDense(output_length, hidden_dim, output_dim, 'TDD')
#Activation
activation = U.Activation('softmax')
#Output layer
output = U.Output()

'''Define Relations'''
encoder.set_input('input_sequence', data, 'output')
decoder.set_input('input_single', encoder, 'output_last')
tdd.set_input('input', decoder, 'output_sequence')
activation.set_input('input', tdd, 'output')
output.set_input('input', activation, 'output')

'''Build Model'''
model = Model()
model.add_input(data, 'X')
model.add_output(output, 'y')
model.add_hidden(encoder)
model.add_hidden(decoder)
model.add_hidden(tdd)
model.add_hidden(activation)


model.compile(optimizer='adam', loss_configures = [('y', 'categorical_crossentropy', None, False, "categorical"),], verbose=0)

score = model.evaluate(data = {'X': D_X_val, 
                           'y': D_y_val},
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(data = {'X': D_X_train, 
                      'y': D_y_train},
              batch_size=BATCH_SIZE, 
              nb_epoch=1,
              show_accuracy=True, 
              verbose=2,
              validation_data={'X': D_X_val, 
                               'y': D_y_val}
              )
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(D_X_val))
        rowX, rowy = D_X_val[np.array([ind])], D_y_val[np.array([ind])]
        preds = model.predict({'X': rowX}, class_mode = {'y': "categorical"}, verbose=0)['y']
        q = convertor.decode(rowX[0],invert=True)
        correct = convertor.decode(rowy[0])
        guess = convertor.decode(preds[0], calc_argmax=False)
        print('Q', q)
        print('T', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        print('---')
