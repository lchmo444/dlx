'''
@author: Xiang Long
'''
import theano
from theano import tensor as T
import numpy
import dlx.util.theano_utils as TU 
from dlx.unit.core import Unit
from dlx import activations, initializations

class RNN(Unit):
    '''Fully-connected RNN where the output is to fed back to input.
    # Input shape
        input_single: 2D tensor with shape `(nb_samples, input_dim)`. 
        input_sequence: 3D tensor with shape `(nb_samples, input_length, input_dim)`.
        input_mask: 2D tensor with shape `(nb_samples, input_length)`.

    # Output shape
        output_sequence: 3D tensor with shape `(nb_samples, input_length, output_dim)`.
        output: 2D tensor with shape `(nb_samples, output_dim)`.
    '''

    def __init__(self, input_length, input_dim, output_dim, name='RNN', truncate_gradient=-1, go_backwards=False,
                 weight_init = 'glorot_uniform', inner_init = 'orthogonal', bias_init = 'zero', activation='sigmoid'):
        super(RNN, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name=name
        self.truncate_gradient = truncate_gradient
        self.go_backwards = go_backwards
        
        self.required_input_sets = [['input_single'], ['input_sequence'], ['input_sequence', 'input_mask']]
        self.output_names = ['output_last', 'output_sequence']
        self.required_function_sets = [['weight_init', 'inner_init', 'bias_init', 'activation']]
        self.set_output('output_last', self.output_last)
        self.set_output('output_sequence', self.output_sequence)
        self.set_function('activation', activations.get(activation))
        self.set_function('weight_init', initializations.get(weight_init))
        self.set_function('inner_init', initializations.get(weight_init))
        self.set_function('bias_init', initializations.get(bias_init))
        
    def build(self):
        self.W = TU.shared(self.get_function('weight_init')((self.input_dim, self.output_dim)), name=self.name+'_W')
        self.U = TU.shared(self.get_function('inner_init')((self.output_dim, self.output_dim)), name=self.name+'_U')
        self.b = TU.shared(self.get_function('bias_init')((self.output_dim,)), name=self.name+'_b')
        self.params = [self.W, self.U, self.b]
    
    def step_single(self, h_tm1, x_h, U):
        h_t = self.get_function('activation')(T.dot(h_tm1, U) + x_h)
        return h_t
    
    def step_sequence(self, x_t, h_tm1, W, U, b):
        h_t = self.get_function('activation')(T.dot(x_t, W) + T.dot(h_tm1, U) + b)
        return h_t
    
    def step_sequence_with_mask(self, mask_t, x_t, h_tm1, W, U, b):
        h_t = self.step_sequence(x_t, h_tm1, W, U, b)
        # mask
        h_t = T.switch(mask_t, h_t, 0. * h_t)
        return h_t       
    
    def output_h_vals(self, train=False):
        if self.inputs_dict.has_key('input_single'):
            x = self.get_input('input_single')(train) #(nb_sample, input_dim)
            h_0 = T.zeros((x.shape[0], self.output_dim))  # (nb_samples, output_dim)
            x_h = T.dot(x, self.W) + self.b
            h_vals, _ = theano.scan(self.step_single,
                                    outputs_info=h_0,
                                    non_sequences=[x_h, self.U],
                                    n_steps=self.input_length,
                                    truncate_gradient=self.truncate_gradient,
                                    go_backwards=self.go_backwards,
                                    strict=True) 
            return h_vals         
            
        else :          
            X = self.get_input('input_sequence')(train) # (nb_sample, input_length, input_dim)
            X = X.dimshuffle((1, 0, 2))  # (input_length, nb_sample, input_dim) 
            h_0 = T.zeros((X.shape[1], self.output_dim))  # (nb_samples, output_dim)
            
            if self.inputs_dict.has_key('input_mask'):
                mask = self.get_input('input_mask')(train) # (nb_sample, input_length)
                mask = T.cast(mask, dtype='int8').dimshuffle((1, 0, 'x')) # (input_length, nb_sample, 1)                
                h_vals, _ = theano.scan(self.step_sequence_with_mask,
                                        sequences=[mask, X],
                                        outputs_info=h_0,
                                        non_sequences=[self.W, self.U, self.b],
                                        truncate_gradient=self.truncate_gradient,
                                        go_backwards=self.go_backwards,
                                        strict=True)
                return h_vals
            else:
                h_vals, _ = theano.scan(self.step_sequence,
                                        sequences=[X],
                                        outputs_info=h_0,
                                        non_sequences=[self.W, self.U, self.b],
                                        truncate_gradient=self.truncate_gradient,
                                        go_backwards=self.go_backwards,
                                        strict=True) 
                return h_vals           
    
    def output_sequence(self, train=False):
        return self.output_h_vals(train).dimshuffle((1, 0, 2)) # (nb_sample, input_length, output_dim)
    
    def output_last(self, train=False):
        return self.output_h_vals(train)[-1] # (nb_sample, output_dim)
    
    
    
class LSTM(Unit):
    '''Long-Short Term Memory.
    # Input shape
        input_single: 2D tensor with shape `(nb_samples, input_dim)`. 
        input_sequences: 3D tensor with shape `(nb_samples, input_length, input_dim)`.
        mask: 2D tensor with shape `(nb_samples, input_length)`.

    # Output shape
        output_sequences: 3D tensor with shape `(nb_samples, input_length, output_dim)`.
        output: 2D tensor with shape `(nb_samples, output_dim)`.
    '''

    def __init__(self, input_length, input_dim, output_dim, name='LSTM', truncate_gradient=-1, go_backwards=False,
                 weight_init = 'glorot_uniform', inner_init = 'orthogonal', bias_init = 'zero', forget_bias_init = 'one',
                 activation='tanh', attention_activation='tanh', inner_activation='hard_sigmoid'):
        super(LSTM, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name=name
        self.truncate_gradient = truncate_gradient
        self.go_backwards = go_backwards
        
        self.required_input_sets = [['input_single'], ['input_sequence'], ['input_sequence', 'input_mask']]
        self.output_names = ['output_last', 'output_sequence']
        self.required_function_sets = [['weight_init', 'inner_init', 'bias_init', 'forget_bias_init', 'activation']]
        self.set_output('output_last', self.output_last)
        self.set_output('output_sequence', self.output_sequence)
        self.set_function('activation', activations.get(activation))
        self.set_function('inner_activation', activations.get(inner_activation))
        self.set_function('weight_init', initializations.get(weight_init))
        self.set_function('inner_init', initializations.get(weight_init))
        self.set_function('bias_init', initializations.get(bias_init))
        self.set_function('forget_bias_init', initializations.get(forget_bias_init))
        
        
    def build(self):
        f_init = self.get_function('weight_init')
        f_inner_init = self.get_function('inner_init')
        f_bias_init = self.get_function('bias_init')
        f_forget_bias_init = self.get_function('forget_bias_init')
         
        _W_i = f_init((self.input_dim, self.output_dim))
        _U_i = f_inner_init((self.output_dim, self.output_dim))
        _b_i = f_bias_init((self.output_dim,))

        _W_f = f_init((self.input_dim, self.output_dim))
        _U_f = f_inner_init((self.output_dim, self.output_dim))
        _b_f = f_forget_bias_init((self.output_dim,))

        _W_c = f_init((self.input_dim, self.output_dim))
        _U_c = f_inner_init((self.output_dim, self.output_dim))
        _b_c = f_bias_init((self.output_dim,))

        _W_o = f_init((self.input_dim, self.output_dim))
        _U_o = f_inner_init((self.output_dim, self.output_dim))
        _b_o = f_bias_init((self.output_dim,),)
        
        self.W = TU.shared(numpy.concatenate([_W_i, _W_f, _W_c, _W_o], axis=1), name=self.name + '_W')
        self.U = TU.shared(numpy.concatenate([_U_i, _U_f, _U_c, _U_o], axis=1), name=self.name + '_U')
        self.b = TU.shared(numpy.concatenate([_b_i, _b_f, _b_c, _b_o]), name=self.name + '_b')
        self.params = [self.W, self.U, self.b]
       
    def _slice(self, P, i, dim):
        return P[:, i*dim:(i+1)*dim]

    def step_single(self, h_tm1, c_tm1, x_h, U):
        f_activation = self.get_function('activation')
        f_inner_activation = self.get_function('inner_activation')
        preact = T.dot(h_tm1, U) + x_h

        i = f_inner_activation(self._slice(preact, 0, self.output_dim))
        f = f_inner_activation(self._slice(preact, 1, self.output_dim))
        c = f * c_tm1 + i * f_activation(self._slice(preact, 3, self.output_dim))
        o = f_inner_activation(self._slice(preact, 2, self.output_dim))

        h = o * f_activation(c)
        return h, c  

    def step_sequence(self, x_t, h_tm1, c_tm1, W, U, b):
        f_activation = self.get_function('activation')
        f_inner_activation = self.get_function('inner_activation')
        preact = T.dot(x_t, W) + T.dot(h_tm1, U) + b

        i = f_inner_activation(self._slice(preact, 0, self.output_dim))
        f = f_inner_activation(self._slice(preact, 1, self.output_dim))
        c = f * c_tm1 + i * f_activation(self._slice(preact, 3, self.output_dim))
        o = f_inner_activation(self._slice(preact, 2, self.output_dim))

        h = o * f_activation(c)
        return h, c  
    
    def step_sequence_with_mask(self, mask_t, x_t, h_tm1, c_tm1, W, U, b):
        h, c = self.step_sequence(self, x_t, h_tm1, c_tm1, W, U, b)
        # mask
        h = T.switch(mask_t, h, 0. * h)
        c = T.switch(mask_t, c, 0. * c)
        return h, c    
       
    
    def output_h_vals(self, train=False):
        if self.inputs_dict.has_key('input_single'):
            x = self.get_input('input_single')(train) #(nb_sample, input_dim)
            h_0 = T.zeros((x.shape[0], self.output_dim))  # (nb_samples, output_dim)
            c_0 = T.zeros((x.shape[0], self.output_dim))  # (nb_samples, output_dim)
            x_h = T.dot(x, self.W) + self.b
            revals, _ = theano.scan(self.step_single,
                                    outputs_info=[h_0, c_0],
                                    non_sequences=[x_h, self.U],
                                    n_steps=self.input_length,
                                    truncate_gradient=self.truncate_gradient,
                                    go_backwards=self.go_backwards,
                                    strict=True) 
            return revals[0]        
            
        else :          
            X = self.get_input('input_sequence')(train) # (nb_sample, input_length, input_dim)
            X = X.dimshuffle((1, 0, 2))  # (input_length, nb_sample, input_dim) 
            h_0 = T.zeros((X.shape[1], self.output_dim))  # (nb_samples, output_dim)
            c_0 = T.zeros((X.shape[1], self.output_dim))  # (nb_samples, output_dim)
            if self.inputs_dict.has_key('input_mask'):
                mask = self.get_input('input_mask')(train) # (nb_sample, input_length)
                mask = T.cast(mask, dtype='int8').dimshuffle((1, 0, 'x')) # (input_length, nb_sample, 1)                
                revals, _ = theano.scan(self.step_sequence_with_mask,
                                        sequences=[mask, X],
                                        outputs_info=[h_0, c_0],
                                        non_sequences=[self.W, self.U, self.b],
                                        truncate_gradient=self.truncate_gradient,
                                        go_backwards=self.go_backwards,
                                        strict=True)
                return revals[0] 
            else:
                revals, _ = theano.scan(self.step_sequence,
                                        sequences=[X],
                                        outputs_info=[h_0, c_0],
                                        non_sequences=[self.W, self.U, self.b],
                                        truncate_gradient=self.truncate_gradient,
                                        go_backwards=self.go_backwards,
                                        strict=True) 
                return revals[0]            
    
    def output_sequence(self, train=False):
        return self.output_h_vals(train).dimshuffle((1, 0, 2)) # (nb_sample, input_length, output_dim)
    
    def output_last(self, train=False):
        return self.output_h_vals(train)[-1] # (nb_sample, output_dim)
    
    