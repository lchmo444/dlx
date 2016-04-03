from dlx import objectives, optimizers
from dlx.util import keras_utils as K
from dlx.util import theano_utils as TU
from dlx.util.generic_utils import Progbar
from dlx import callbacks as cbks
import numpy as np

def standardize_y(y):
    if not hasattr(y, 'shape'):
        y = np.asarray(y)
    if len(y.shape) == 1:
        y = np.expand_dims(y, 1)
    return y


def batch_shuffle(index_array, batch_size):
    batch_count = int(len(index_array) / batch_size)
    # to reshape we need to be cleanly divisible by batch size
    # we stash extra items and reappend them after shuffling
    last_batch = index_array[batch_count * batch_size:]
    index_array = index_array[:batch_count * batch_size]
    index_array = index_array.reshape((batch_count, batch_size))
    np.random.shuffle(index_array)
    index_array = index_array.flatten()
    return np.append(index_array, last_batch)


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]


def slice_X(X, start=None, stop=None):
    '''
    '''
    if type(X) == list:
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return X[start]
        else:
            return X[start:stop]
        
def categorical_accuracy(y_true, y_pred):
    return K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))

def binary_accuracy(y_true, y_pred):
    return K.equal(y_true, K.round(y_pred))
        
def standardize_weights(y, sample_weight=None, class_weight=None):
    '''
    '''
    if sample_weight is not None:
        assert len(sample_weight) == len(y)
        return sample_weight.flatten()
    elif isinstance(class_weight, dict):
        if len(y.shape) > 2:
            raise Exception('class_weight not supported for '
                            '3+ dimensional targets.')
        if y.shape[1] > 1:
            y_classes = y.argmax(axis=1)
        elif y.shape[1] == 1:
            y_classes = np.reshape(y, y.shape[0])
        else:
            y_classes = y
        weights = np.asarray([class_weight[cls] for cls in y_classes])
        return weights
    else:
        return np.ones((y.shape[0],))


def standardize_l(l):
    if type(l) == list:
        return l
    elif type(l) == tuple:
        return list[l]
    else:
        return [l]


def weighted_objective(fn):
    def weighted(y_true, y_pred, weights, mask=None):
        # score_array has ndim >= 2
        score_array = fn(y_true, y_pred)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            mask = K.cast(mask, K.floatX)
            # mask should have the same shape as score_array
            score_array *= mask
            #  the loss per batch should be proportional
            #  to the number of unmasked samples.
            score_array /= K.mean(mask)

        # reduce score_array to 1D
        ndim = K.ndim(score_array)
        for _ in range(ndim-1):
            score_array = K.mean(score_array, axis=-1)

        if weights is not None:
            score_array *= weights
        return K.mean(score_array)
    return weighted

class Model(object):
    def __init__(self):
        self.input_units = []
        self.output_units = []
        self.hidden_units = []
        
        self.input_names = []
        self.output_names = []
    
    def add_input(self, unit, name):
        self.input_units.append(unit)
        self.input_names.append(name)
    
    def add_output(self, unit, name_list):
        self.output_units.append(unit)
        self.output_names.append(standardize_l(name_list))
    
    def add_hidden(self, unit):
        self.hidden_units.append(unit)
        
    def compile(self, optimizer, loss_configures, verbose = 0):
        '''Configure the learning process.

        # Arguments
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](optimizers.md).
            loss_configures: list of tuples: (output name, objective function, mask name, weighted, class_mode) 
                weighted: true or false. If true, must provide class_weight or sample_weight in same order.                        
                objective function can be string name of objective function or function or None(for output without calculate loss).
                class_mode: one of "categorical", "binary", None.
        '''
        self.optimizer = optimizers.get(optimizer)
        self.loss_configures = loss_configures
        self.input_order = self.input_names
        self.train_output_order = []
        self.predict_output_order = []
        self.weight_order = []
        self.out_labels = ['loss',]
        
        if verbose:
            print 'input units:', self.input_units
            print 'output units:', self.output_units
            print 'hidden units:', self.hidden_units
        
        units = self.input_units + self.output_units + self.hidden_units
        for unit in units:
            unit.check()
        for unit in units:
            unit.build()
            
        self.params = []
        self.regularizers = []    
        self.constraints = []
        self.updates = []    
        for unit in self.hidden_units:
            pars = unit.get_params()
            self.params += pars[0]
            self.regularizers += pars[1]
            self.constraints += pars[2]
            self.updates += pars[3]
        
        if verbose:
            print 'parameters:', [TU.sp(param) for param in self.params]
            print 'regularizers:', self.regularizers
            print 'constraints:', self.constraints
            print 'updates:', self.updates
         
        outputs_train = []
        outputs_test = []   
        ys_test = [] 
        for output_unit, output_name_list in zip(self.output_units, self.output_names): 
            res_train = standardize_l(output_unit.get_results(train=True))
            if len(output_name_list) != len(res_train):
                raise Exception('Number of outputs(train) not match number of names (%s)'%str(output_name_list))
            res_test = standardize_l(output_unit.get_results(train=False))
            if len(output_name_list) != len(res_test):
                raise Exception('Number of outputs(test) not match number of names (%s)'%str(output_name_list))  
                      
            outputs_train.append(res_train)
            outputs_test.append(res_test)
            ys_test += res_test
            self.predict_output_order += output_name_list
            
        if verbose:
            print 'output names:', self.output_names
            print 'outputs train:', outputs_train
            print 'outputs test:', outputs_test
            if verbose >= 2:
                for vares, names in zip(outputs_train, self.output_names):
                    for var,name in zip(vares, names):
                        if verbose == 2:
                            print 'output_' + name + ':', TU.sp(var)
                        else:
                            print 'output_' + name, ':'
                            TU.dp(var)
            
            
        def find_output(output_name, train):
            if train:
                outputs = outputs_train
            else:
                outputs = outputs_test
            
            for output_list, output_name_list in zip(outputs, self.output_names): 
                for output, name in zip(output_list, output_name_list):
                    if output_name == name:
                        return output
            raise Exception('Can not find output "%s".'%output_name)
        
        ys = []
        ys_train = []
        train_accs = []
        test_accs = []
        weights = []
        train_loss = 0.
        test_loss = 0.
        
        for loss_cf in loss_configures:
              
            output_name = loss_cf[0]
            obj_fn = loss_cf[1]
            mask_name = loss_cf[2]
            weighted = loss_cf[3]
            class_mode = loss_cf[4]
            
            y_train = find_output(output_name, train=True)
            y_test = find_output(output_name ,train=False)
            
            if mask_name:
                mask_train = find_output(mask_name, train=True)
                mask_test = find_output(mask_name, train=False)
            else:
                mask_train = None
                mask_test = None
            
            y = TU.tensor(ndim=K.ndim(y_train), name=output_name)
            ys.append(y)
            
            ys_train.append(y_train)
            self.train_output_order.append(output_name)
            
            if obj_fn:
                
                if weighted:
                    self.weight_order += output_name
                    weight = TU.tensor(1, name=output_name+'_weight')                   
                    weights.append(weight)           
                    weighted_loss = weighted_objective(objectives.get(obj_fn))
                    train_loss += weighted_loss(y, y_train, weight, mask_train)
                    test_loss += weighted_loss(y, y_test, weight, mask_test)
                else:
                    weighted_loss = weighted_objective(objectives.get(obj_fn))
                    train_loss += weighted_loss(y, y_train, None, mask_train)
                    test_loss += weighted_loss(y, y_test, None, mask_test)        
                
            if class_mode:
                self.out_labels.append('acc_'+output_name)
                
                if class_mode == "categorical":
                    weighted_accuracy = weighted_objective(categorical_accuracy)
                    train_accuracy = weighted_accuracy(y, y_train, None, mask_train)
                    test_accuracy = weighted_accuracy(y, y_test, None, mask_test)
                    
                elif class_mode == "binary":
                    weighted_accuracy = weighted_objective(binary_accuracy)
                    train_accuracy = weighted_accuracy(y, y_train, None, mask_train)
                    test_accuracy = weighted_accuracy(y, y_test, None, mask_test)
                else:
                    raise Exception("Invalid class mode:" + str(class_mode))
                train_accs.append(train_accuracy)
                test_accs.append(test_accuracy)

        if verbose:
            print 'ys:', ys
            print 'ys_train:', ys_train
            print 'predict output order:', self.predict_output_order
            print 'ys_test:', ys_test
            print 'train output order:', self.train_output_order
            print 'train_accs:', train_accs
            print 'test_accs:', test_accs
            print 'weight:', weights  
        if verbose == 2:
            print 'train_loss:', TU.sp(train_loss)
            print 'test_loss:', TU.sp(test_loss)
        if verbose >= 3:
            print 'train_loss:'
            TU.dp(train_loss)
            print 'test_loss:' 
            TU.dp(test_loss)
            
  
                        
        ins = []
        for input_unit in self.input_units:
            ins.append(input_unit.get_variable())
        
        if verbose:
            print 'ins:', ins
           
        train_vars = ins + ys + weights
        test_vars = ins + ys + weights
        
        for r in self.regularizers:
            train_loss = r(train_loss)

        updates = self.optimizer.get_updates(self.params,
                                             self.constraints,
                                             train_loss)
        
        state_updates = self.updates
            
        updates += state_updates
        
        if verbose:
            print 'train_vars:', train_vars
            print 'test_vars:', test_vars

        if verbose:
            print 'updates:'
            for update in updates:
                print update       
        
        self._train = K.function(train_vars, [train_loss], updates=updates)
        self._train_with_acc = K.function(train_vars, [train_loss] + train_accs, updates=updates)
        self._test = K.function(test_vars, [test_loss], updates=state_updates)
        self._test_with_acc = K.function(test_vars, [test_loss] + test_accs, updates=state_updates)
        self._predict = K.function(inputs=ins, outputs=ys_test, updates=state_updates)
        
        
        
        
    def fit(self, data, batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            show_accuracy=False, class_weight={}, sample_weight={}):
        '''Train the model for a fixed number of epochs.

        Returns a history object. Its `history` attribute is a record of
        training loss values at successive epochs,
        as well as validation loss values (if applicable).

        # Arguments
            data: dictionary mapping input names and outputs names to
                appropriate numpy arrays. All arrays should contain
                the same number of samples.
            batch_size: int. Number of samples per gradient update.
            nb_epoch: int.
            verbose: 0 for no logging to stdout,
                1 for progress bar logging, 2 for one log line per epoch.
            callbacks: `keras.callbacks.Callback` list. List of callbacks
                to apply during training. See [callbacks](callbacks.md).
            validation_split: float (0. < x < 1). Fraction of the data to
                use as held-out validation data.
            validation_data: dictionary mapping input names and outputs names
                to appropriate numpy arrays to be used as
                held-out validation data.
                All arrays should contain the same number of samples.
                Will override validation_split.
            shuffle: boolean. Whether to shuffle the samples at each epoch.
            class_weight: dictionary mapping output names to
                class weight dictionaries.
            sample_weight: dictionary mapping output names to
                numpy arrays of sample weights.
        '''
        X = [data[name] for name in self.input_order]
        y = [standardize_y(data[name]) for name in self.train_output_order]
        if len(set([len(a) for a in X] + [len(a) for a in y])) != 1:
            raise Exception('All input arrays and target arrays must have '
                            'the same number of samples.')

        sample_weight_list = [standardize_weights(y[i], sample_weight=sample_weight.get(self.weight_order[i])) for i in range(len(self.weight_order))]
        class_weight_list = [class_weight.get(name) for name in self.weight_order]

        val_f = None
        val_ins = None
        if validation_data or validation_split:
            if show_accuracy:
                val_f = self._test_with_acc
            else:
                val_f = self._test
        if validation_data:
            # can't use sample weights with validation data at this point
            y_val = [standardize_y(validation_data[name]) for name in self.train_output_order]
            sample_weight = [standardize_weights(y_val[i]) for i in range(len(self.weight_order))]
            val_ins = [validation_data[name] for name in self.input_order] + [standardize_y(validation_data[name]) for name in self.train_output_order] + sample_weight

        elif 0 < validation_split < 1:
            split_at = int(len(X[0]) * (1 - validation_split))
            X, X_val = (slice_X(X, 0, split_at), slice_X(X, split_at))
            y, y_val = (slice_X(y, 0, split_at), slice_X(y, split_at))
            sample_weight_list, sample_weight_list_val = (slice_X(sample_weight_list, 0, split_at), slice_X(sample_weight_list, split_at))
            val_ins = X_val + y_val + sample_weight_list_val

        if show_accuracy:
            f = self._train_with_acc
            out_labels = self.out_labels
        else:
            f = self._train
            out_labels = ['loss']
        
        metrics = ['loss']
        for label in out_labels:
            if label.startswith('acc_'):
                metrics.append(label)
                
        metrics.append('val_loss')
        for label in out_labels:
            if label.startswith('acc_'):
                metrics.append('val_'+label)

        sample_weight_list = [standardize_weights(y[i],
                                                  sample_weight=sample_weight_list[i],
                                                  class_weight=class_weight_list[i]) for i in range(len(self.weight_order))]
        ins = X + y + sample_weight_list
        history = self._fit(f, ins, out_labels=out_labels,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=verbose, callbacks=callbacks,
                            val_f=val_f, val_ins=val_ins,
                            shuffle=shuffle, metrics=metrics)
        return history 
        
        
    def evaluate(self, data, batch_size=128, show_accuracy=False, verbose=0, sample_weight={}):
        '''Compute the loss on some input data, batch by batch.

        Arguments: see `fit` method.
        '''
        sample_weight = [standardize_weights(data[name],
                                             sample_weight=sample_weight.get(name)) for name in self.weight_order]
        ins = [data[name] for name in self.input_order] + [standardize_y(data[name]) for name in self.train_output_order] + sample_weight
        if len(set([len(a) for a in ins])) != 1:
            raise Exception('All input arrays and target arrays must have '
                            'the same number of samples.')
        
        
        if show_accuracy:
            f = self._test_with_acc
        else:
            f = self._test
            
        outs = self._test_loop(f, ins, batch_size, verbose)
        
        if show_accuracy:
            return outs
        else:
            return outs[0]
        
        
    def predict(self, data, class_mode={}, batch_size=128, verbose=0):
        '''Generate output predictions for the input samples
        batch by batch.

        Arguments: see `fit` method.
        class_mode: dict {output_name: one of "categorical", "binary", None}
        '''
        ins = [data[name] for name in self.input_order]
        modes = [class_mode[name] for name in self.predict_output_order]
        
        if len(set([len(a) for a in ins])) != 1:
            raise Exception('All input arrays and target arrays must have '
                            'the same number of samples.')
        outs = []
        probas = self._predict_loop(self._predict, ins, batch_size, verbose)

        for proba, mode in zip(probas, modes):
            if mode:
                if mode == 'categorical':
                    outs.append(proba.argmax(axis=-1))
                else:
                    outs.append((proba > 0.5).astype('int32'))  
            else:
                outs.append(proba)         

        return dict(zip(self.predict_output_order, outs))       
        
    
    def get_weights(self):
        weights = []
        for unit in self.hidden_units:
            weights += unit.get_weights()
        return weights

    def set_weights(self, weights):
        for unit in self.hidden_units:
            nb_param = len(unit.get_weights())
            unit.set_weights(weights[:nb_param])
            weights = weights[nb_param:]
            
            
    def save_weights(self, filepath, overwrite=False):
        '''Save weights from all layers to a HDF5 files.
        '''
        import h5py
        import os.path
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(filepath):
            import sys
            get_input = input
            if sys.version_info[:2] <= (2, 7):
                get_input = raw_input
            overwrite = get_input('[WARNING] %s already exists - overwrite? '
                                  '[y/n]' % (filepath))
            while overwrite not in ['y', 'n']:
                overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
            if overwrite == 'n':
                return
            print('[TIP] Next time specify overwrite=True in save_weights!')

        f = h5py.File(filepath, 'w')
        g = f.create_group('model')
        weights = self.get_weights()
        g.attrs['nb_params'] = len(weights)
        for n, param in enumerate(weights):
            param_name = 'param_{}'.format(n)
            param_dset = g.create_dataset(param_name, param.shape,
                                          dtype=param.dtype)
            param_dset[:] = param
        f.flush()
        f.close()

    def load_weights(self, filepath):
        '''Load weights from a HDF5 file.
        '''
        import h5py
        f = h5py.File(filepath)
        g = f['model']
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        self.set_weights(weights)
        f.close()
    
    
    '''Abstract base model class.
    '''
    def _fit(self, f, ins, out_labels=[], batch_size=128,
             nb_epoch=100, verbose=1, callbacks=[],
             val_f=None, val_ins=None, shuffle=True, metrics=[]):
        '''
            Abstract fit function for f(ins).
            Assume that f returns a list, labelled by out_labels.
        '''
        self.training_data = ins
        self.validation_data = val_ins
        do_validation = False
        if val_f and val_ins:
            do_validation = True
            if verbose:
                print('Train on %d samples, validate on %d samples' %
                      (len(ins[0]), len(val_ins[0])))

        nb_train_sample = len(ins[0])
        index_array = np.arange(nb_train_sample)

        history = cbks.History()
        if verbose:
            callbacks = [history, cbks.BaseLogger()] + callbacks
        else:
            callbacks = [history] + callbacks
        callbacks = cbks.CallbackList(callbacks)

        callbacks._set_model(self)
        callbacks._set_params({
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': metrics,
        })
        callbacks.on_train_begin()

        self.stop_training = False
        for epoch in range(nb_epoch):
            callbacks.on_epoch_begin(epoch)
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    ins_batch = slice_X(ins, batch_ids)
                except TypeError:
                    raise Exception('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = f(ins_batch)
                if type(outs) != list:
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                epoch_logs = {}
                if batch_index == len(batches) - 1:  # last batch
                    # validation
                    if do_validation:
                        # replace with self._evaluate
                        val_outs = self._test_loop(val_f, val_ins,
                                                   batch_size=batch_size,
                                                   verbose=0)
                        if type(val_outs) != list:
                            val_outs = [val_outs]
                        # same labels assumed
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        return history

    def _predict_loop(self, f, ins, batch_size=128, verbose=0):
        '''Abstract method to loop over some data in batches.
        '''
        nb_sample = len(ins[0])
        outs = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_ids)

            batch_outs = f(ins_batch)
            if type(batch_outs) != list:
                batch_outs = [batch_outs]
            if batch_index == 0:
                for batch_out in batch_outs:
                    shape = (nb_sample,) + batch_out.shape[1:]
                    outs.append(np.zeros(shape))

            for i, batch_out in enumerate(batch_outs):
                outs[i][batch_start:batch_end] = batch_out
            if verbose == 1:
                progbar.update(batch_end)
        return outs

    def _test_loop(self, f, ins, batch_size=128, verbose=0):
        '''Abstract method to loop over some data in batches.
        '''
        nb_sample = len(ins[0])
        outs = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_ids)

            batch_outs = f(ins_batch)
            if type(batch_outs) == list:
                if batch_index == 0:
                    for batch_out in enumerate(batch_outs):
                        outs.append(0.)
                for i, batch_out in enumerate(batch_outs):
                    outs[i] += batch_out * len(batch_ids)
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += batch_outs * len(batch_ids)

            if verbose == 1:
                progbar.update(batch_end)
        for i, _ in enumerate(outs):
            outs[i] /= nb_sample
        return outs
        
        
        
        
        
        
        
        
        
    