import theano
from theano import tensor as T 
import numpy as np
from dlx.config import floatX

def sp(x):
    return theano.printing.pprint(x)

def pp(x):
    print theano.printing.pprint(x)

def dp(x):
    print theano.printing.debugprint(x)

def tensor(ndim, name=None, dtype=floatX):
    '''Instantiate an theano variable.
    '''
    if ndim == 0:
        return T.scalar(name=name, dtype=dtype)
    elif ndim == 1:
        return T.vector(name=name, dtype=dtype)
    elif ndim == 2:
        return T.matrix(name=name, dtype=dtype)
    elif ndim == 3:
        return T.tensor3(name=name, dtype=dtype)
    elif ndim == 4:
        return T.tensor4(name=name, dtype=dtype)
    else:
        raise Exception('ndim too large: ' + str(ndim))
    
    
def shared(value, dtype=floatX, name=None):
    '''Instantiate a  variable.
    '''
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, strict=False)

def repeat(x, n):
    '''
    # Input shape
        2D tensor of shape `(d1, d2)`.
    # Output shape
        3D tensor of shape `(n, d1, d2)`.
    '''
    tensors = [x] * n
    stacked = T.stack(*tensors)
    return stacked