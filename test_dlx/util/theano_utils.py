import dlx.util.theano_utils as TU
import theano.printing as P

def _test_tensor():
    print '\n------------------------------------------------------------'
    print 'Test: dlx.util.theano_utils.tensor'
    x0 = TU.tensor(0, 'x0')
    x1 = TU.tensor(1, 'x1')
    x2 = TU.tensor(2, 'x2')
    x3 = TU.tensor(3, 'x3')
    x4 = TU.tensor(4, 'x4')
    print x0, type(x0)
    print x1, type(x1)
    print x2, type(x2)
    print x3, type(x3)
    print x4, type(x4)

def _test_repeat():
    print '\n------------------------------------------------------------'
    print 'Test: dlx.util.theano_utils.repeat'
    x2 = TU.tensor(2, 'x2')
    x3 = TU.tensor(3, 'x3')
    y3 = TU.repeat(x2, 10)
    y4 = TU.repeat(x3, 10)
    print P.pprint(y3)
    print P.pprint(y4)


_test_tensor()
_test_repeat()