# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from data.big_number_data_engine import BigNumberDataEngine
from data.character_data_engine import CharacterDataEngine
from dlx.model import slice_X
import sys
np.random.seed(1337)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset
TRAINING_SIZE = 200000
MIN_DIGITS = 10
DIGITS = 12
INVERT = True
HIDDEN_SIZE = 128
BATCH_SIZE = 256
LAYERS = 1
MAXLEN = DIGITS + 1

print('Generating data...')
engine = BigNumberDataEngine(min_digits=12, max_digits=DIGITS)
As, Bs, expected = engine.get_seperate_dataset(TRAINING_SIZE)
print('Total additions:', len(As))

print('Vectorization...')
convertor = CharacterDataEngine(engine.get_character_set(), maxlen=MAXLEN, soldier = ' ')
D_A = convertor.encode_dataset(As, invert=True, index = True)
D_B = convertor.encode_dataset(Bs, invert=True, index = True)
D_y = convertor.encode_dataset(expected, maxlen=MAXLEN ,invert=True)

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
indices = np.arange(len(D_y))
np.random.shuffle(indices)
D_A = D_A[indices]
D_B = D_B[indices]
D_y = D_y[indices]

# Explicitly set apart 10% for validation data that we never train over
split_at = len(D_A) - len(D_A) / 10
(D_A_train, D_A_val) = (slice_X(D_A, 0, split_at), slice_X(D_A, split_at))
(D_B_train, D_B_val) = (slice_X(D_B, 0, split_at), slice_X(D_B, split_at))
(D_y_train, D_y_val) = (D_y[:split_at], D_y[split_at:])

print(D_A_train.shape)
print(D_B_train.shape)
print(D_y_train.shape)


import dlx.unit.core as U
import dlx.unit.attention as A
from dlx.model import Model

print('Build model...')
input_dim = convertor.get_dim() + MAXLEN
output_dim = convertor.get_dim()
hidden_dim = HIDDEN_SIZE
output_length = MAXLEN
attention_hidden_dim = HIDDEN_SIZE

'''Define Units'''
#Data unit
dataA = U.Input(3, 'A')
dataB = U.Input(3, 'B')
#Add Remove 1
add1 = U.AddOneAtBegin()
remove1 = U.RemoveOneAtBegin()
#Attention
decoder = A.AttentionLSTM_X(output_length+1, input_dim, hidden_dim, input_dim, attention_hidden_dim, name='ATT')
#One to Many
one2many = U.OneToMany(['y', 'alpha'])
#Time Distributed Dense
tdd = U.TimeDistributedDense(output_length, hidden_dim, output_dim, 'TDD')
#Activation
activation = U.Activation('softmax')
#Output layer
output_y = U.Output()
output_alpha = U.Output()

'''Define Relations'''
add1.set_input('input', dataA, 'output')
decoder.set_input('input_sequence', add1, 'output')
decoder.set_input('context', dataB, 'output')
one2many.set_input('input', decoder, 'output_sequence_with_alpha')
remove1.set_input('input', one2many, 'y')
tdd.set_input('input', remove1, 'output')
activation.set_input('input', tdd, 'output')
output_y.set_input('input', activation, 'output')
output_alpha.set_input('input', one2many, 'alpha')

'''Build Model'''
model = Model()
model.add_input(dataA, 'A')
model.add_input(dataB, 'B')
model.add_output(output_y, ['y'])
model.add_output(output_alpha, ['alpha'])
model.add_hidden(decoder)
model.add_hidden(one2many)
model.add_hidden(add1)
model.add_hidden(remove1)
model.add_hidden(tdd)
model.add_hidden(activation)


model.compile(optimizer='adam', loss_configures = [('y', 'categorical_crossentropy', None, False, "categorical"),], verbose=0)

score = model.evaluate(data = {'A': D_A_val,
                               'B': D_B_val, 
                           'y': D_y_val},
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 1000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    history = model.fit(data = {'A': D_A_train,
                      'B': D_B_train,  
                      'y': D_y_train},
              batch_size=BATCH_SIZE, 
              nb_epoch=1,
              show_accuracy=True, 
              verbose=2,
              validation_data={'A': D_A_val,
                               'B': D_B_val,  
                               'y': D_y_val}
              )
    
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for ii in range(10):
        ind = np.random.randint(0, len(D_A_val))
        rowA, rowB, rowy = D_A_val[np.array([ind])], D_B_val[np.array([ind])], D_y_val[np.array([ind])]
        preds = model.predict({'A': rowA, 'B': rowB}, class_mode = {'y': "categorical", 'alpha': None}, verbose=0)
        string_A = convertor.decode(rowA[0],invert=True, index = True)
        string_B = convertor.decode(rowB[0],invert=True, index = True)
        correct = convertor.decode(rowy[0],invert=True)
        guess = convertor.decode(preds['y'][0], calc_argmax=False)[::-1]
        alpha = preds['alpha']
        
        print('A', string_A)
        print(colors.ok + '+' + colors.close)
        print('B', string_B)
        print(colors.ok + '=' + colors.close)
        print('T', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        
        if ii == 9:
            print('Alpha:', alpha[0].shape)
            sys.stdout.write('GUESS    A\\B')
            for j in range(output_length):
                sys.stdout.write("%6s"% string_B[j])
            sys.stdout.write("\n")
            for i in range(output_length+1):
                if i == output_length:
                    sys.stdout.write("%5s%6s"% ('#','#'))
                else:
                    sys.stdout.write("%5s%6s"% (guess[i], string_A[i]))
                for j in range(output_length):
                    sys.stdout.write("%6.2f"% alpha[0, output_length-i, output_length-1-j])
                sys.stdout.write("\n")
            
        print('-------------------------------')
        
    print('Iteration', iteration)   
    print('loss:', history.history['loss'][0],
          ' - accuracy:', history.history['acc_y'][0],
          ' - val_loss:', history.history['val_loss'][0],
          ' - val_accuracy:', history.history['val_acc_y'][0],
         )   
    