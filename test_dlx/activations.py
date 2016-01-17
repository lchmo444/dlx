from dlx import activations
from dlx.config import floatX
import numpy as np
import theano
import theano.tensor as T

print '\n------------------------------------------------------------'
print 'Test, dlx.activations'

data2 = np.asarray(np.random.uniform(size=[3, 5]), dtype=floatX)
data3 = np.asarray(np.random.uniform(size=(2, 3, 5)), dtype=floatX)

print 'data2,'
print data2
print 'data3,'
print data3

def add_one(x):
    return x + 1.0

init_list = [('softmax', data2),
             ('softplus', data2),
             ('relu', data2),
             ('tanh', data2),
             ('sigmoid',data2),
             ('hard_sigmoid',data2),
             ('linear',data2),
             (add_one, data2),
             ('softmax', data3),
             ('softplus', data3),
             ('relu', data3),
             ('tanh', data3),
             ('sigmoid',data3),
             ('hard_sigmoid',data3),
             ('linear',data3),
             (add_one, data3)
            ]

for (act, data) in init_list:
    if data.ndim == 2:
        x = T.matrix('x')
    else:
        x = T.tensor3('x')
    fun = theano.function([x], activations.get(act)(x))
    val = fun(data)
    print act, 'data' + str(data.ndim), ':'
    print val
    print