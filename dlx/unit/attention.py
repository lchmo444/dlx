'''
@author: Xiang Long
'''
import theano
from theano import tensor as T
import numpy
import dlx.util.theano_utils as TU 
from dlx.unit.core import Unit
from dlx import activations, initializations

class AttentionLSTM(Unit):
    '''Long-Short Term Memory with soft Attention.
    # Input shape
        input_single: 2D tensor with shape `(nb_samples, input_dim)`. 
        input_sequences: 3D tensor with shape `(nb_samples, input_length, input_dim)`.
        mask: 2D tensor with shape `(nb_samples, input_length)`.

    # Output shape
        output_sequences: 3D tensor with shape `(nb_samples, input_length, output_dim)`.
        output: 2D tensor with shape `(nb_samples, output_dim)`.
    '''

    def __init__(self, input_length, input_dim, output_dim, context_dim, attention_hidden_dim, 
                 name='AttentionLSTM', truncate_gradient=-1, go_backwards=False,
                 weight_init = 'glorot_uniform', inner_init = 'orthogonal', bias_init = 'zero', forget_bias_init = 'one',
                 activation='tanh', attention_activation='tanh', inner_activation='hard_sigmoid'):
        super(AttentionLSTM, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.name=name
        self.truncate_gradient = truncate_gradient
        self.go_backwards = go_backwards
        
        self.required_input_sets = [['input_single', 'context'], ['input_sequence', 'context'], ['input_sequence', 'input_mask', 'context']]
        self.output_names = ['output_last', 'output_sequence', 'output_sequence_with_alpha', 'output_last_with_alpha']
        self.required_function_sets = [['weight_init', 'inner_init', 'bias_init', 'forget_bias_init', 'activation', 'attention_activation']]
        self.set_output('output_last', self.output_last)
        self.set_output('output_sequence', self.output_sequence)
        self.set_output('output_last_with_alpha', self.output_last_with_alpha)
        self.set_output('output_sequence_with_alpha', self.output_sequence_with_alpha)
        self.set_function('activation', activations.get(activation))
        self.set_function('attention_activation', activations.get(attention_activation))
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
        
        '''
        Attention hidden dense and projector
        '''
        self.W_h_att = TU.shared(f_init((self.output_dim, self.attention_hidden_dim)), name=self.name + '_W_h_att')
        self.W_ctx_att = TU.shared(f_init((self.context_dim, self.attention_hidden_dim)), name=self.name + '_W_ctx_att')
        self.b_att =  TU.shared(f_bias_init((self.attention_hidden_dim,)), name=self.name + '_b_att')
        self.w_att_prj = TU.shared(f_init((self.attention_hidden_dim, 1)), name=self.name + '_w_att_prj')

        ''' 
        LSTM {W: x, U: h, V: weighted context}
        '''
        W_i = f_init((self.input_dim, self.output_dim))
        V_i = f_init((self.context_dim, self.output_dim))
        U_i = f_inner_init((self.output_dim, self.output_dim))
        b_i = f_bias_init((self.output_dim,))

        W_f = f_init((self.input_dim, self.output_dim))
        V_f = f_init((self.context_dim, self.output_dim))
        U_f = f_inner_init((self.output_dim, self.output_dim))
        b_f = f_forget_bias_init((self.output_dim,))

        W_c = f_init((self.input_dim, self.output_dim))
        V_c = f_init((self.context_dim, self.output_dim))
        U_c = f_inner_init((self.output_dim, self.output_dim))
        b_c = f_bias_init((self.output_dim,))

        W_o = f_init((self.input_dim, self.output_dim))
        V_o = f_init((self.context_dim, self.output_dim))
        U_o = f_inner_init((self.output_dim, self.output_dim))
        b_o = f_bias_init((self.output_dim,))
        
        # theano variables
        self.W = TU.shared(numpy.concatenate([W_i, W_f, W_c, W_o], axis=1), name=self.name + '_W')
        self.V = TU.shared(numpy.concatenate([V_i, V_f, V_c, V_o], axis=1), name=self.name + '_V')
        self.U = TU.shared(numpy.concatenate([U_i, U_f, U_c, U_o], axis=1), name=self.name + '_U')
        self.b = TU.shared(numpy.concatenate([b_i, b_f, b_c, b_o]), name=self.name + '_b')
        
        self.params = [self.W, self.V, self.U, self.b, self.W_h_att, self.W_ctx_att, self.b_att, self.w_att_prj]
    
    def _slice(self, P, i, dim):
        return P[:, i*dim:(i+1)*dim]
    
    def step_single(self, h_tm1, c_tm1, x_h, ctx, att_ctx, V, U, W_h_att, w_att_prj):
        f_activation = self.get_function('activation')
        f_inner_activation = self.get_function('inner_activation')
        f_attention_activation = self.get_function('attention_activation')
        
        # attention
        h_att = T.dot(h_tm1, W_h_att) # (nb_sample, input_dim) dot (input_dim, attention_hidden_dim) -> (nb_sample, attention_hidden_dim)
        preprj = f_attention_activation(h_att[:,None,:] + att_ctx) #(nb_sample, nb_context, attention_hidden_dim)
        #prj_ctx = T.flatten(T.dot(preprj, w_att_prj), 2)  #(nb_sample, nb_context)
        prj_ctx = T.dot(preprj, w_att_prj).reshape((x_h.shape[0], ctx.shape[1]))  #(nb_sample, nb_context)
        alpha = T.nnet.softmax(prj_ctx) #(nb_sample, nb_context)
        weighted_ctx = (ctx * alpha[:,:,None]).sum(1) # (nb_sample, context_dim)
        
        # LSTM
        preact = T.dot(weighted_ctx, V) + T.dot(h_tm1, U) + x_h
        i = f_inner_activation(self._slice(preact, 0, self.output_dim))
        f = f_inner_activation(self._slice(preact, 1, self.output_dim))
        c = f * c_tm1 + i * f_activation(self._slice(preact, 3, self.output_dim))
        o = f_inner_activation(self._slice(preact, 2, self.output_dim))
        h = o * f_activation(c)
        return h, c, alpha    
    
    def step_sequence(self, x_t, h_tm1, c_tm1, ctx, att_ctx, W, V, U, b, W_h_att, w_att_prj):
        f_activation = self.get_function('activation')
        f_inner_activation = self.get_function('inner_activation')
        f_attention_activation = self.get_function('attention_activation')
        
        # attention
        h_att = T.dot(h_tm1, W_h_att) # (nb_sample, input_dim) dot (input_dim, attention_hidden_dim) -> (nb_sample, attention_hidden_dim)
        preprj = f_attention_activation(h_att[:,None,:] + att_ctx) #(nb_sample, nb_context, attention_hidden_dim)
        #prj_ctx = T.flatten(T.dot(preprj, w_att_prj), 2)  #(nb_sample, nb_context)
        prj_ctx = T.dot(preprj, w_att_prj).reshape((x_t.shape[0], ctx.shape[1]))  #(nb_sample, nb_context)
        alpha = T.nnet.softmax(prj_ctx) #(nb_sample, nb_context)
        weighted_ctx = (ctx * alpha[:,:,None]).sum(1) # (nb_sample, context_dim)
        
        # LSTM
        preact = T.dot(x_t, W) + T.dot(weighted_ctx, V) + T.dot(h_tm1, U) + b
        i = f_inner_activation(self._slice(preact, 0, self.output_dim))
        f = f_inner_activation(self._slice(preact, 1, self.output_dim))
        c = f * c_tm1 + i * f_activation(self._slice(preact, 3, self.output_dim))
        o = f_inner_activation(self._slice(preact, 2, self.output_dim))
        h = o * f_activation(c)
        return h, c, alpha
    
    def step_sequence_with_mask(self, mask_t, x_t, h_tm1, c_tm1, ctx, att_ctx, W, V, U, b, W_h_att, w_att_prj):
        h, c, alpha = self.step_sequence(x_t, h_tm1, c_tm1, ctx, att_ctx, W, V, U, b, W_h_att, w_att_prj)
        # mask
        h = T.switch(mask_t, h, 0. * h)
        c = T.switch(mask_t, c, 0. * c)
        return h, c, alpha    

    def output_revals(self, train=False):
        context = self.get_input('context')(train)  # (nb_samples, nb_context, context_dim)
        att_ctx = T.dot(context, self.W_ctx_att) + self.b_att  # (nb_samples, nb_context, attention_hidden_dim) + (attention_hidden_dim,)
        
        if self.inputs_dict.has_key('input_single'):
            x = self.get_input('input_single')(train) #(nb_sample, input_dim)
            h_0 = T.zeros((x.shape[0], self.output_dim))  # (nb_samples, output_dim)
            c_0 = T.zeros((x.shape[0], self.output_dim))  # (nb_samples, output_dim)
            x_h = T.dot(x, self.W) + self.b
            
            revals, _ = theano.scan(self.step_single,
                                    outputs_info=[h_0, c_0, None],
                                    non_sequences=[x_h, context, att_ctx, self.V, self.U, self.W_h_att, self.w_att_prj],
                                    n_steps=self.input_length,
                                    truncate_gradient=self.truncate_gradient,
                                    go_backwards=self.go_backwards,
                                    strict=True) 
            return revals        
            
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
                                        outputs_info=[h_0, c_0, None],
                                        non_sequences=[context, att_ctx, self.W, self.V, self.U, self.b, self.W_h_att, self.w_att_prj],
                                        truncate_gradient=self.truncate_gradient,
                                        go_backwards=self.go_backwards,
                                        strict=True)
                return revals
            else:
                revals, _ = theano.scan(self.step_sequence,
                                        sequences=[X],
                                        outputs_info=[h_0, c_0, None],
                                        non_sequences=[context, att_ctx, self.W, self.V, self.U, self.b, self.W_h_att, self.w_att_prj],
                                        truncate_gradient=self.truncate_gradient,
                                        go_backwards=self.go_backwards,
                                        strict=True)
                return revals                 
    
    def output_sequence(self, train=False):
        return self.output_revals(train)[0].dimshuffle((1, 0, 2)) # (nb_sample, input_length, output_dim)
    
    def output_last(self, train=False):
        return self.output_revals(train)[0][-1] # (nb_sample, output_dim)
    
    def output_sequence_with_alpha(self, train=False):
        revals = self.output_revals(train)  #(h, c, alpha)
        return [revals[0].dimshuffle((1, 0, 2)), # (nb_sample, input_length, output_dim)
                revals[2].dimshuffle((1, 0, 2))] # (nb_sample, input_length, nb_context) 
    
    def output_last_with_alpha(self, train=False):
        revals = self.output_revals(train)  #(h, c, alpha)
        return [revals[0][-1], # (nb_sample, output_dim)
                revals[2][-1]] # (nb_sample, nb_context) 
    
    
class AttentionLSTM_X(Unit):
    '''Long-Short Term Memory with soft Attention.
    # Input shape
        input_single: 2D tensor with shape `(nb_samples, input_dim)`. 
        input_sequences: 3D tensor with shape `(nb_samples, input_length, input_dim)`.
        mask: 2D tensor with shape `(nb_samples, input_length)`.

    # Output shape
        output_sequences: 3D tensor with shape `(nb_samples, input_length, output_dim)`.
        output: 2D tensor with shape `(nb_samples, output_dim)`.
    '''

    def __init__(self, input_length, input_dim, output_dim, context_dim, attention_hidden_dim, 
                 name='AttentionLSTM', truncate_gradient=-1, go_backwards=False,
                 weight_init = 'glorot_uniform', inner_init = 'orthogonal', bias_init = 'zero', forget_bias_init = 'one',
                 activation='tanh', attention_activation='tanh', inner_activation='hard_sigmoid'):
        super(AttentionLSTM_X, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.name=name
        self.truncate_gradient = truncate_gradient
        self.go_backwards = go_backwards
        
        self.required_input_sets = [['input_single', 'context'], ['input_sequence', 'context'], ['input_sequence', 'input_mask', 'context']]
        self.output_names = ['output_last', 'output_sequence', 'output_sequence_with_alpha', 'output_last_with_alpha']
        self.required_function_sets = [['weight_init', 'inner_init', 'bias_init', 'forget_bias_init', 'activation', 'attention_activation']]
        self.set_output('output_last', self.output_last)
        self.set_output('output_sequence', self.output_sequence)
        self.set_output('output_last_with_alpha', self.output_last_with_alpha)
        self.set_output('output_sequence_with_alpha', self.output_sequence_with_alpha)
        self.set_function('activation', activations.get(activation))
        self.set_function('attention_activation', activations.get(attention_activation))
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
        
        '''
        Attention hidden dense and projector
        '''
        self.W_h_att = TU.shared(f_init((self.output_dim, self.attention_hidden_dim)), name=self.name + '_W_h_att')
        self.W_x_att = TU.shared(f_init((self.input_dim, self.attention_hidden_dim)), name=self.name + '_W_x_att')
        self.W_ctx_att = TU.shared(f_init((self.context_dim, self.attention_hidden_dim)), name=self.name + '_W_ctx_att')
        self.b_att =  TU.shared(f_bias_init((self.attention_hidden_dim,)), name=self.name + '_b_att')
        self.w_att_prj = TU.shared(f_init((self.attention_hidden_dim, 1)), name=self.name + '_w_att_prj')

        ''' 
        LSTM {W: x, U: h, V: weighted context}
        '''
        W_i = f_init((self.input_dim, self.output_dim))
        V_i = f_init((self.context_dim, self.output_dim))
        U_i = f_inner_init((self.output_dim, self.output_dim))
        b_i = f_bias_init((self.output_dim,))

        W_f = f_init((self.input_dim, self.output_dim))
        V_f = f_init((self.context_dim, self.output_dim))
        U_f = f_inner_init((self.output_dim, self.output_dim))
        b_f = f_forget_bias_init((self.output_dim,))

        W_c = f_init((self.input_dim, self.output_dim))
        V_c = f_init((self.context_dim, self.output_dim))
        U_c = f_inner_init((self.output_dim, self.output_dim))
        b_c = f_bias_init((self.output_dim,))

        W_o = f_init((self.input_dim, self.output_dim))
        V_o = f_init((self.context_dim, self.output_dim))
        U_o = f_inner_init((self.output_dim, self.output_dim))
        b_o = f_bias_init((self.output_dim,))
        
        # theano variables
        self.W = TU.shared(numpy.concatenate([W_i, W_f, W_c, W_o], axis=1), name=self.name + '_W')
        self.V = TU.shared(numpy.concatenate([V_i, V_f, V_c, V_o], axis=1), name=self.name + '_V')
        self.U = TU.shared(numpy.concatenate([U_i, U_f, U_c, U_o], axis=1), name=self.name + '_U')
        self.b = TU.shared(numpy.concatenate([b_i, b_f, b_c, b_o]), name=self.name + '_b')
        
        self.params = [self.W, self.V, self.U, self.b, self.W_h_att, self.W_ctx_att, self.b_att, self.w_att_prj]
    
    def _slice(self, P, i, dim):
        return P[:, i*dim:(i+1)*dim]
    
    def step_single(self, h_tm1, c_tm1, x_h, x_att, ctx, att_ctx, V, U, W_h_att, w_att_prj):
        f_activation = self.get_function('activation')
        f_inner_activation = self.get_function('inner_activation')
        f_attention_activation = self.get_function('attention_activation')
        
        # attention
        h_att = T.dot(h_tm1, W_h_att) # (nb_sample, input_dim) dot (input_dim, attention_hidden_dim) -> (nb_sample, attention_hidden_dim)
        preprj = f_attention_activation(h_att[:,None,:] + x_att[:,None,:] + att_ctx) #(nb_sample, nb_context, attention_hidden_dim)
        #prj_ctx = T.flatten(T.dot(preprj, w_att_prj), 2)  #(nb_sample, nb_context)
        prj_ctx = T.dot(preprj, w_att_prj).reshape((x_h.shape[0], ctx.shape[1]))  #(nb_sample, nb_context)
        alpha = T.nnet.softmax(prj_ctx) #(nb_sample, nb_context)
        weighted_ctx = (ctx * alpha[:,:,None]).sum(1) # (nb_sample, context_dim)
        
        # LSTM
        preact = T.dot(weighted_ctx, V) + T.dot(h_tm1, U) + x_h
        i = f_inner_activation(self._slice(preact, 0, self.output_dim))
        f = f_inner_activation(self._slice(preact, 1, self.output_dim))
        c = f * c_tm1 + i * f_activation(self._slice(preact, 3, self.output_dim))
        o = f_inner_activation(self._slice(preact, 2, self.output_dim))
        h = o * f_activation(c)
        return h, c, alpha    
    
    def step_sequence(self, x_t, h_tm1, c_tm1, ctx, att_ctx, W, V, U, b, W_h_att, W_x_att, w_att_prj):
        f_activation = self.get_function('activation')
        f_inner_activation = self.get_function('inner_activation')
        f_attention_activation = self.get_function('attention_activation')
        
        # attention
        h_att = T.dot(h_tm1, W_h_att) # (nb_sample, hidden_dim) dot (hidden_dim, attention_hidden_dim) -> (nb_sample, attention_hidden_dim)
        x_att = T.dot(x_t, W_x_att)  # (nb_sample, input_dim) dot (input_dim, attention_hidden_dim) -> (nb_sample, attention_hidden_dim)
        preprj = f_attention_activation(h_att[:,None,:] + x_att[:,None,:] + att_ctx) #(nb_sample, nb_context, attention_hidden_dim)
        #prj_ctx = T.flatten(T.dot(preprj, w_att_prj), 2)  #(nb_sample, nb_context)
        prj_ctx = T.dot(preprj, w_att_prj).reshape((x_t.shape[0], ctx.shape[1]))  #(nb_sample, nb_context)
        alpha = T.nnet.softmax(prj_ctx) #(nb_sample, nb_context)
        weighted_ctx = (ctx * alpha[:,:,None]).sum(1) # (nb_sample, context_dim)
        
        # LSTM
        preact = T.dot(x_t, W) + T.dot(weighted_ctx, V) + T.dot(h_tm1, U) + b
        i = f_inner_activation(self._slice(preact, 0, self.output_dim))
        f = f_inner_activation(self._slice(preact, 1, self.output_dim))
        c = f * c_tm1 + i * f_activation(self._slice(preact, 3, self.output_dim))
        o = f_inner_activation(self._slice(preact, 2, self.output_dim))
        h = o * f_activation(c)
        return h, c, alpha
    
    def step_sequence_with_mask(self, mask_t, x_t, h_tm1, c_tm1, ctx, att_ctx, W, V, U, b, W_h_att, W_x_att, w_att_prj):
        h, c, alpha = self.step_sequence(x_t, h_tm1, c_tm1, ctx, att_ctx, W, V, U, b, W_h_att, W_x_att, w_att_prj)
        # mask
        h = T.switch(mask_t, h, 0. * h)
        c = T.switch(mask_t, c, 0. * c)
        return h, c, alpha    

    def output_revals(self, train=False):
        context = self.get_input('context')(train)  # (nb_samples, nb_context, context_dim)
        att_ctx = T.dot(context, self.W_ctx_att) + self.b_att  # (nb_samples, nb_context, attention_hidden_dim) + (attention_hidden_dim,)
        
        if self.inputs_dict.has_key('input_single'):
            x = self.get_input('input_single')(train) #(nb_sample, input_dim)
            h_0 = T.zeros((x.shape[0], self.output_dim))  # (nb_samples, output_dim)
            c_0 = T.zeros((x.shape[0], self.output_dim))  # (nb_samples, output_dim)
            x_h = T.dot(x, self.W) + self.b
            x_att = T.dot(x, self.W_x_att)
            
            revals, _ = theano.scan(self.step_single,
                                    outputs_info=[h_0, c_0, None],
                                    non_sequences=[x_h, x_att, context, att_ctx, self.V, self.U, self.W_h_att, self.w_att_prj],
                                    n_steps=self.input_length,
                                    truncate_gradient=self.truncate_gradient,
                                    go_backwards=self.go_backwards,
                                    strict=True) 
            return revals        
            
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
                                        outputs_info=[h_0, c_0, None],
                                        non_sequences=[context, att_ctx, self.W, self.V, self.U, self.b, self.W_h_att, self.W_x_att, self.w_att_prj],
                                        truncate_gradient=self.truncate_gradient,
                                        go_backwards=self.go_backwards,
                                        strict=True)
                return revals
            else:
                revals, _ = theano.scan(self.step_sequence,
                                        sequences=[X],
                                        outputs_info=[h_0, c_0, None],
                                        non_sequences=[context, att_ctx, self.W, self.V, self.U, self.b, self.W_h_att, self.W_x_att, self.w_att_prj],
                                        truncate_gradient=self.truncate_gradient,
                                        go_backwards=self.go_backwards,
                                        strict=True)
                return revals                 
    
    def output_sequence(self, train=False):
        return self.output_revals(train)[0].dimshuffle((1, 0, 2)) # (nb_sample, input_length, output_dim)
    
    def output_last(self, train=False):
        return self.output_revals(train)[0][-1] # (nb_sample, output_dim)
    
    def output_sequence_with_alpha(self, train=False):
        revals = self.output_revals(train)  #(h, c, alpha)
        return [revals[0].dimshuffle((1, 0, 2)), # (nb_sample, input_length, output_dim)
                revals[2].dimshuffle((1, 0, 2))] # (nb_sample, input_length, nb_context) 
    
    def output_last_with_alpha(self, train=False):
        revals = self.output_revals(train)  #(h, c, alpha)
        return [revals[0][-1], # (nb_sample, output_dim)
                revals[2][-1]] # (nb_sample, nb_context) 