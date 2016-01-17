import dlx.unit.core as U
import theano.printing as P

def _test_Input():
    print '\n------------------------------------------------------------'
    print 'Test: Input Unit'
    data1 = U.Input(2, name='Data1')
    data1.check()
    data1.build()
    print P.pprint(data1.get_variable())
    data2 = U.Input(2)
    data2.check()
    data2.build()
    print P.pprint(data1.get_variable())
 
def _test_Mask():
    print '\n------------------------------------------------------------' 
    print 'Test: Mask Unit'
    x = U.Input(3, name='X')
    mask = U.Mask()
    mask.set_input('input', x, 'output')
    output = U.Output()
    output.set_input('input', mask, 'mask')
    x.build()
    mask.check()
    mask.build()
    output.build()
    print P.pprint(output.get_results(train=False))
   
def _test_Dropout():
    print '\n------------------------------------------------------------'
    print 'Test: Dropout Unit'
    data_1 = U.Input(2, name='X')
    dropout = U.Dropout(0.2)
    dropout.set_input('input', data_1, 'output')
    data_1.build()
    dropout.check()
    dropout.build()
    print P.pprint(dropout.get_output('output')(train=False))  
    print P.pprint(dropout.get_output('output')(train=True))
    
def _test_Activation():
    print '\n------------------------------------------------------------' 
    print 'Test: Activation Unit'
    x = U.Input(2, name='X')
    relu = U.Activation('relu')
    relu.set_input('input', x, 'output')
    softmax = U.Activation('softmax')
    softmax.set_input('input', x, 'output')
    x.build()
    relu.check()
    relu.build()
    softmax.check()
    softmax.build()
    print P.pprint(relu.get_output('output')(train=False))
    print P.pprint(softmax.get_output('output')(train=False))

def _test_Output():
    print '\n------------------------------------------------------------' 
    print 'Test: Output Unit'
    x = U.Input(2, name='X')
    output = U.Output()
    output.set_input('input', x, 'output')
    x.build()
    output.check()
    output.build()
    print P.pprint(output.get_results(train=False))
    
    
def _test_Lambda():
    print '\n------------------------------------------------------------' 
    print 'Test: Lambda Unit'
    x = U.Input(2, name='X')
    y = U.Input(2, name='Y')
    def fun(x, y):
        return x*2, x+y, y*2
    f = U.Lambda(fun, ['2x', 'x+y', '2y'])
    f.set_input('input_x', x, 'output')
    f.set_input('input_y', y, 'output')
    x.build()
    y.build()
    f.check()
    f.build()
    print P.pprint(f.get_output('2x')(train=False))
    print P.pprint(f.get_output('x+y')(train=False))
    print P.pprint(f.get_output('2y')(train=False))

    output1 = U.Output()
    output1.set_input('input', f, '2x')
    output1.build()
    print P.pprint(output1.get_results(train=False))
    
def _test_SimpleLambda():
    print '\n------------------------------------------------------------' 
    print 'Test: Simple Lambda Unit'
    x = U.Input(2, name='X')
    def fun(x):
        return x**2
    f = U.SimpleLambda(fun)
    f.set_input('input', x, 'output')
    x.build()
    f.check()
    f.build()
    print P.pprint(f.get_output('output')(train=False))
    
def _test_RepeatVector():
    print '\n------------------------------------------------------------' 
    print 'Test: Repeat Vector Unit'
    x = U.Input(2, name='X')
    f = U.RepeatVector(10)
    f.set_input('input', x, 'output')
    x.build()
    f.check()
    f.build()
    print P.pprint(f.get_output('output')(train=False))

   
def _test_Dense():
    print '\n------------------------------------------------------------'
    print 'Test: Dense Unit'
    X = U.Input(2, name='X')
    dense_1 = U.Dense(16,24, name='Dense1')
    dense_1.set_input('input', X, 'output')
    X.build()
    dense_1.check()
    dense_1.build()
    print P.pprint(dense_1.get_output('output')(train=False))

def _test_TimeDistributedDense():
    print '\n------------------------------------------------------------' 
    print 'Test: Time Distributed Dense Unit'
    x = U.Input(3, name='X')
    tdd = U.TimeDistributedDense(16,1024,128)
    tdd.set_input('input', x, 'output')
    x.build()
    tdd.check()
    tdd.build()
    print P.debugprint(tdd.get_output('output')(train=False))

if __name__ == '__main__':
    _test_Input()
    _test_Mask()
    _test_Dense()
    _test_Dropout()
    _test_Activation()
    _test_Output()
    _test_Lambda()
    _test_SimpleLambda()
    _test_RepeatVector()
    _test_TimeDistributedDense()
