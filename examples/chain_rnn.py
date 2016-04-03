# -*- coding: utf-8 -*-
from __future__ import print_function
from data.simple_chain_engine import SimpleChainEngine
from tool.convertor import word_one_hot_vector_convertor

import numpy as np
np.random.seed(1334)
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

def sentence2str(sentence):
    s = ''
    for word in sentence:
        s += word + ' '
    return s
    

words = [str(i) for i in range(10)] + [chr(i) for i in range(65,75)]
print ('words:', words)

#Dataset
DATA_SIZE = 1000
HIDDEN_SIZE = 128
BATCH_SIZE = 33
MAXLEN = len(words)

engine = SimpleChainEngine(words)
starts, sentences = engine.get_dataset(DATA_SIZE)
for (i, start, sentence) in zip(range(DATA_SIZE), starts, sentences):
    print ("%s -> %s" %(sentence2str(start), sentence2str(sentence)))
    if i>=5:
        break

sinputs = [sentence[:-1] for sentence in sentences]
soutputs = [sentence[1:] for sentence in sentences]
for (i, sinput, soutput) in zip(range(DATA_SIZE), sinputs, soutputs):
    print ("%s -> %s" %(sentence2str(sinput), sentence2str(soutput)))
    if i>=5:
        break

convertor = word_one_hot_vector_convertor(engine.get_dictionary())
D_X, D_mask = convertor.sentences2one_hot_tensor(sinputs, MAXLEN)
D_Y, _ = convertor.sentences2one_hot_tensor(soutputs, MAXLEN)
print (D_X.shape, D_Y.shape, D_mask.shape)

# Shuffle (X, Y)
indices = np.arange(DATA_SIZE)
np.random.shuffle(indices)
D_X = D_X[indices]
D_Y = D_Y[indices]
D_mask = D_mask[indices]

# Explicitly set apart 10% for validation data that we never train over
split_at = DATA_SIZE - DATA_SIZE / 10
(D_X_train, D_X_val) = (D_X[:split_at], D_X[split_at:])
(D_Y_train, D_Y_val) = (D_Y[:split_at], D_Y[split_at:])
(D_mask_train, D_mask_val) = (D_mask[:split_at], D_mask[split_at:])
print (D_X_train.shape, D_Y_train.shape, D_mask_train.shape)


import dlx.unit.core as U
import dlx.unit.recurrent as R
from dlx.model import Model

print('Build model...')
input_dim = output_dim = len(engine.get_dictionary())
hidden_dim = HIDDEN_SIZE
input_length = MAXLEN
output_length = MAXLEN

'''Define Units'''
#Data unit
data = U.Input(3, 'X')
#Mask unit
mask_in = U.Input(2, 'MASK_IN')
#RNN
rnn = R.RNN(input_length, input_dim, hidden_dim, name='RNN')
#Time Distributed Dense
tdd = U.TimeDistributedDense(output_length, hidden_dim, output_dim, 'TDD')
#Activation
activation = U.Activation('softmax')
#Output layer
output = U.Output()
#Output layer
mask_out = U.Output()

'''Define Relations'''
rnn.set_input('input_sequence', data, 'output')
rnn.set_input('input_mask', mask_in, 'output')
tdd.set_input('input', rnn, 'output_sequence')
activation.set_input('input', tdd, 'output')
output.set_input('input', activation, 'output')
mask_out.set_input('input', mask_in, 'output')

'''Build Model'''
model = Model()
model.add_input(data, 'X')
model.add_output(output, 'Y')
model.add_input(mask_in, 'MASK_IN')
model.add_output(mask_out, 'MASK_OUT')
model.add_hidden(rnn)
model.add_hidden(tdd)
model.add_hidden(activation)

model.compile(optimizer='adam', loss_configures = [('Y', 'categorical_crossentropy', 'MASK_OUT', False, "categorical"),], verbose=0)

score = model.evaluate(data = {'X': D_X_val,
                               'MASK_IN': D_mask_val, 
                               'Y': D_Y_val},
                       batch_size=BATCH_SIZE, 
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(data = {'X': D_X_train, 
                      'MASK_IN': D_mask_train,
                      'Y': D_Y_train},
              batch_size=BATCH_SIZE, 
              nb_epoch=1,
              show_accuracy=True, 
              verbose=2,
              validation_data={'X': D_X_val,
                               'MASK_IN': D_mask_val, 
                               'Y': D_Y_val}
              )
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(5):
        ind = np.random.randint(0, len(D_X_val))
        rowX, rowY, rowMask = D_X_val[np.array([ind])], D_Y_val[np.array([ind])], D_mask_val[np.array([ind])]
        res = model.predict({'X': rowX, 'MASK_IN': rowMask}, class_mode = {'Y': None, 'MASK_OUT': None}, verbose=0)
        preds = res['Y']
        masks = res['MASK_OUT']
        sinput = convertor.one_hot_matrix2sentence(rowX[0], rowMask[0])
        soutput = convertor.one_hot_matrix2sentence(rowY[0], rowMask[0])
        spredict = convertor.one_hot_matrix2sentence(preds[0], masks[0])
        print('I', sentence2str(sinput))
        print('O', sentence2str(spredict))
        print(colors.ok + '☑' + colors.close if spredict == soutput else colors.fail + '☒' + colors.close, sentence2str(soutput))
        print('---')



















