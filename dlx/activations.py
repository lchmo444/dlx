import theano
import theano.tensor as T


'''
Activation functions for 'theano variable' (not numpy array)
# Input:
    x: theano variable
# Output:
    out: theano variable

'''

def softmax(x):  
    ndim = x.ndim
    if ndim == 2:
        return T.nnet.softmax(x)
    elif ndim == 3:
        # x (nb_samples, timesteps, dim)
        # apply softmax to each timestep
        def step(x):
            return T.nnet.softmax(x)
        x = x.dimshuffle((1,0,2))
        x, _ = theano.scan(step, sequences=x, outputs_info=None)
        return x.dimshuffle((1,0,2))
    else:
        raise Exception('Cannot apply softmax to a tensor that is not 2D or 3D. ' +
                        'Here, ndim=' + str(ndim))


def softplus(x):
    return T.nnet.softplus(x)


def relu(x, alpha=0., max_value=None):
    x = T.nnet.relu(x, alpha)
    if max_value is not None:
        x = T.minimum(x, max_value)
    return x


def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


def linear(x):
    '''
    The function returns the variable that is passed in, so all types work.
    '''
    return x


from dlx.util.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'activation function')
