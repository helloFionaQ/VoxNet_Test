import numpy as np
import theano
a=np.arange(60).reshape(3,4,5,1)
print a,a.shape
print '................................'
x=a.reshape(3,4,1,5)
print x,x.shape
print '................................'
x=a.dimshuffle(0,1,3,2)
print x,x.shape