'''
@author: Xiang Long
'''
import theano
import theano.tensor as T
import dlx.util.theano_utils as TU
from dlx import initializations, activations
from dlx import constraints
from dlx.util import keras_utils as K
from __builtin__ import False

class Unit(object):
    '''
    Abstract Unit
    '''
    def __init__(self):
        '''
        Define inputs_dict, outputs_dict, functions_dict
        '''
        self.inputs_dict = {}
        self.outputs_dict = {}
        self.functions_dict = {}
        
        self.required_input_sets = [[]]
        self.required_function_sets = [[]]
        self.output_names = []
        
        self.params = []
        self.constraints = []
        self.constraint = None # for all parameters
        self.regularizers = []
        self.updates = []   
        
        self.cache = {}
        
    def set_input(self, input_name, input_Unit, output_name):
        self.inputs_dict[input_name] = (input_Unit, output_name)
        
    def set_output(self, output_name, output_function):
        self.outputs_dict[output_name] = output_function 
    
    def set_function(self, function_name, function):
        self.functions_dict[function_name] = function
    
    def get_input(self, input_name):
        ''' out: function '''
        (input_Unit, output_name) = self.inputs_dict[input_name]
        return input_Unit.get_output(output_name)
    
    def cache_output_fun(self, output, output_name):
        def cache_output(train=False):
            cache_id = "%s_%s" %(output_name, train)         
            if not self.cache.has_key(cache_id):
                self.cache[cache_id] = output(train)
            return self.cache[cache_id]
        return cache_output
    
    def get_output(self, output_name):
        ''' out: function '''
        return self.cache_output_fun(self.outputs_dict[output_name], output_name)
    
    def get_function(self, function_name):
        ''' out: function '''
        return self.functions_dict[function_name]
    
    def get_required_input_sets(self):
        return self.required_input_sets

    def get_output_names(self):
        return self.output_names
    
    def get_required_function_sets(self):
        return self.required_function_sets
    
            
    def check_set(self, _set, _dict):
        for funs in _set: 
            flag = True    
            for fun in funs:
                if not _dict.has_key(fun):
                    flag = False
                    break;
            if flag:
                return True
        return False
        
    def check(self):
        # check output (almost defined inside the Unit)
        for out in self.output_names:
            if not self.outputs_dict.has_key(out):
                raise Exception('Can not find output %s.%s.' %(self.__class__, out))
        # check function
        if not self.check_set(self.required_function_sets, self.functions_dict):
            raise Exception('Can not find suitable function set. Possible function sets of %s: %s' %(self.__class__, str(self.required_function_sets)))
        # check input
        if not self.check_set(self.required_input_sets, self.inputs_dict):
            raise Exception('Can not find suitable input set. Possible input sets of %s: %s' %(self.__class__, str(self.required_input_sets)))  
    
    def build(self):
        pass
    
    def get_params(self):
        # standardize constraints
        consts = []
        if len(self.constraints) == len(self.params):
            for c in self.constraints:
                if c:
                    consts.append(c)
                else:
                    consts.append(constraints.identity())
        elif self.constraint:
            consts += [self.constraint for _ in range(len(self.params))]
        else:
            consts += [constraints.identity() for _ in range(len(self.params))]

        return self.params, self.regularizers, consts, self.updates
    
class Input(Unit):
    '''A Unit for data input.
    # Output shape
        input_dim dimension tensor.
    # Input mask
        default None
    '''
    def __init__(self, input_dim, name='Input'):
        super(Input, self).__init__()
        self.input_dim = input_dim
        self.name = name
        
        self.output_names = ['output']
        self.set_output('output', self.output)

    def build(self):
        self.X = TU.tensor(self.input_dim, name=self.name)
        
    def get_variable(self):
        return self.X

    def output(self, train=False):
        return self.X
    
class Mask(Unit):
    ''' Get mask or a input
    # Input shape
        3D tensor with shape: `(nb_samples, input_length, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, input_length)`.
    '''
    def __init__(self, mask_value=0.):
        super(Mask, self).__init__()
        self.mask_value = mask_value
        
        self.required_input_sets = [['input']]
        self.output_names = ['mask']
        self.set_output('mask', self.mask)
        
    def mask(self, train=False):
        X = self.get_input('input')(train)
        return T.any(T.ones_like(X) * (1. - T.eq(X, self.mask_value)), axis=-1) 
    
    
class Dropout(Unit):
    '''Apply Dropout to the input. Dropout consists in randomly setting
    a fraction `p` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    '''
    def __init__(self, p, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.p = p

        self.required_input_sets = [['input']]
        self.output_names = ['output']
        self.set_output('output', self.output)

    def output(self, train=False):
        X = self.get_input('input')(train)
        if self.p > 0.:
            if train:
                X = K.dropout(X, level=self.p)
        return X

class Activation(Unit):
    '''Apply an activation function to an output.

    # Input shape
        Arbitrary.
    # Output shape
        Same shape as input.
    '''
    def __init__(self, activation):
        super(Activation, self).__init__()
        
        self.required_input_sets = [['input']]
        self.output_names = ['output']
        self.required_function_sets = [['activation']]
        self.set_output('output', self.output)
        self.set_function('activation', activations.get(activation))

    def output(self, train=False):
        X = self.get_input('input')(train)
        return self.get_function('activation')(X)

class Output(Unit):
    '''A Unit for output
    '''
    def __init__(self):
        super(Output, self).__init__()
        
        self.required_input_sets = [['input']]
        
    def get_results(self, train=False):
        return self.get_input('input')(train) 
    
class Lambda(Unit):
    '''A Unit perform a given function without state 
    # required input = ['input_' + arg[0], 'input_' + arg[1]]
    '''
    
    def __init__(self, function, return_names):
        super(Lambda, self).__init__()
        import inspect
        self.arg_names= ['input_' + arg for arg in inspect.getargspec(function)[0]]
        self.return_names = return_names
        
        self.required_input_sets = [self.arg_names]
        self.output_names = self.return_names
        self.required_function_sets = [['function']]
        
        self.set_function('function', function)
        for return_name in return_names:
            self.set_output(return_name, self.get_output_function(return_name))
           
    def get_output_function(self, return_name):
        return lambda train=False:self.output(return_name,train)
    
    def output(self, return_name, train=False):
        outputs = self.get_function('function')(*[self.get_input(arg_name)(train) for arg_name in self.arg_names])
        return outputs[self.return_names.index(return_name)]

        
class SimpleLambda(Unit):
    def __init__(self, function):
        super(SimpleLambda, self).__init__()
        
        self.required_input_sets = [['input']]
        self.output_names += ['output']
        self.required_function_sets += [['function']]
        self.set_output('output', self.output)
        self.set_function('function', function)

    def output(self, train=False):
        X = self.get_input('input')(train)
        return self.get_function('function')(X)    
        
        
class RepeatVector(Unit):
    '''Repeat the input n times.

    # Input shape
        2D tensor of shape `(nb_samples, input_dim)`.

    # Output shape
        3D tensor of shape `(nb_samples, n, input_dim)`.

    # Arguments
        n: integer, repetition factor.
    '''
    def __init__(self, n):
        super(RepeatVector, self).__init__()
        self.n = n
        
        self.required_input_sets = [['input']]
        self.output_names += ['output']
        self.set_output('output', self.output)

    def output(self, train=False):
        X = self.get_input('input')(train)
        return TU.repeat(X, self.n).dimshuffle((1, 0, 2))
  
  
class Dense(Unit):
    '''Fully connected NN Unit.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''
    def __init__(self, input_dim, output_dim, name='Dense', weight_init='glorot_uniform', bias_init='zero', activation='linear'):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name=name
        
        self.required_input_sets = [['input']]
        self.output_names = ['output']
        self.required_function_sets = [['weight_init', 'bias_init', 'activation']]
        self.set_output('output', self.output)
        self.set_function('activation', activations.get(activation))
        self.set_function('weight_init', initializations.get(weight_init))
        self.set_function('bias_init', initializations.get(bias_init))

    def build(self):
        self.W = TU.shared(self.get_function('weight_init')((self.input_dim, self.output_dim)), name=self.name+'_W')
        self.b = TU.shared(self.get_function('bias_init')((self.output_dim,)), name=self.name+'_b')

        self.params = [self.W, self.b]

    def output(self, train=False):
        X = self.get_input('input')(train)
        output = self.get_function('activation')(K.dot(X, self.W) + self.b)
        return output        
  
class TimeDistributedDense(Unit):
    '''Apply a same Dense Unit for each dimension[1] (time dimension) input.
    Especially useful after a recurrent network with 'return_sequence=True'.

    # Input shape
        3D tensor with shape `(nb_sample, input_length, input_dim)`.

    # Output shape
        3D tensor with shape `(nb_sample, input_length, output_dim)`.
    '''
    def __init__(self, input_length, input_dim, output_dim, name='TDD', weight_init='glorot_uniform', bias_init='zero', activation='linear'):
        super(TimeDistributedDense, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name=name
        
        self.required_input_sets = [['input']]
        self.output_names = ['output']
        self.required_function_sets = [['weight_init', 'bias_init', 'activation']]
        self.set_output('output', self.output)
        self.set_function('activation', activations.get(activation))
        self.set_function('weight_init', initializations.get(weight_init))
        self.set_function('bias_init', initializations.get(bias_init))
        

    def build(self):
        self.W = TU.shared(self.get_function('weight_init')((self.input_dim, self.output_dim)), name=self.name+'_W')
        self.b = TU.shared(self.get_function('bias_init')((self.output_dim,)), name=self.name+'_b')

        self.params = [self.W, self.b]

    def output(self, train=False):
        X = self.get_input('input')(train) # (nb_sample, input_length, input_dim)
        X = X.dimshuffle((1, 0, 2))  # (input_length, nb_sample, input_dim) 
        def step(x, W, b):
            output = K.dot(x, W) + b
            return output
        
        outputs, _ = theano.scan(step,
                                sequences= X,
                                outputs_info=[],
                                non_sequences=[self.W, self.b],
                                strict=True) # (input_length, nb_sample, output_dim) 
        outputs = outputs.dimshuffle((1, 0, 2)) # (nb_sample, input_length, output_dim)     
        outputs = self.get_function('activation')(outputs) # (nb_sample, input_length, output_dim)    
        return outputs
    
    
class TimeDistributedMerge(Unit):
    '''Sum/multiply/average over the outputs of a TimeDistributed layer.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        2D tensor with shape: `(samples, features)`.

    # Arguments
        mode: one of {'sum', 'mul', 'ave'}
    '''
    input_ndim = 3

    def __init__(self, mode='sum'):
        super(TimeDistributedMerge, self).__init__()
        self.mode = mode
        
        self.required_input_sets = [['input']]
        self.output_names = ['output']
        self.set_output('output', self.output)

    def output(self, train=False):
        X = self.get_input('input')(train)
        if self.mode == 'ave':
            s = K.mean(X, axis=1)
            return s
        if self.mode == 'sum':
            s = K.sum(X, axis=1)
            return s
        elif self.mode == 'mul':
            s = K.prod(X, axis=1)
            return s
        else:
            raise Exception('Unknown merge mode')



class AddOneAtBegin(Unit):
    def __init__(self, name='ADD1'):
        super(AddOneAtBegin, self).__init__()    
        
        self.required_input_sets = [['input']]
        self.output_names = ['output']
        self.set_output('output', self.output)
        
    def output(self, train=False):
        X = self.get_input('input')(train) # (nb_sample, input_length, input_dim)

        first = T.zeros((X.shape[0], 1, X.shape[2]));
        outputs = T.concatenate([first, X], axis=1)
     
        return outputs

class RemoveOneAtBegin(Unit):
    def __init__(self, first_init='one', name='REMOVE1'):
        super(RemoveOneAtBegin, self).__init__()    
        
        self.required_input_sets = [['input']]
        self.output_names = ['output']
        self.set_output('output', self.output)

    def output(self, train=False):
        X = self.get_input('input')(train) # (nb_sample, input_length, input_dim)

        outputs = X[:,1:,:]
              
        return outputs         
    
class OneToMany(Unit):

    def __init__(self, output_names):
        super(OneToMany, self).__init__()
        
        self.required_input_sets = [['input']]
        self.output_names = output_names
        self.output_catch = {}
        for name in output_names:
            self.set_output(name, self.get_output_func(name))
        
    def get_output_func(self, output_name):
        return lambda train=False:self.output(output_name, train)
        
    def output(self, output_name, train=False):
        index = self.output_names.index(output_name)
        if not self.output_catch.has_key(train):
            self.output_catch[train] = self.get_input('input')(train)
        return self.output_catch[train][index] 
    
    
    
    
    