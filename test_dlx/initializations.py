from dlx import initializations
import numpy as np

print '\n------------------------------------------------------------'
print 'Test: dlx.initializations'

def two(shape):
    return 2. * np.ones(shape)

init_dict = {'uniform': (2,3,4),
             'normal': (2,3,4),
             'lecun_uniform':[2,3,4],
             'glorot_normal':(2,3,4),
             'glorot_uniform':(2,3,4),
             'he_normal':[2,3,4],
             'he_uniform':(2,3,4),
             'orthogonal':(4,4),
             'identity':(4,4),
             'zero':(2,3,4),
             'one':(2,3,4),
             two: (2,3,4)
            }

for fun, shape in init_dict.items():
    val = initializations.get(fun)(shape)
    print fun, shape, val.dtype, ':'
    print val
    print