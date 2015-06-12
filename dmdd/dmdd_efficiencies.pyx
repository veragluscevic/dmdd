import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double tanh(double)
    double sqrt(double)
    double atan2(double,double)
    double acos(double)
    double abs(double)
    double log(double)
    double ceil(double)
    double fabs(double)
    double exp(double)

#################################################
@cython.boundscheck(False)
def efficiency_Xe(np.ndarray[DTYPE_t] Q):
    npts = len(Q)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef unsigned int i
    for i in range(npts):
        out[i]=1.
    return out

#################################################
@cython.boundscheck(False)
def efficiency_Ge(np.ndarray[DTYPE_t] Q):
    npts = len(Q)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef unsigned int i
    for i in range(npts):
        out[i]=1.
    return out
#################################################
@cython.boundscheck(False)
def efficiency_I(np.ndarray[DTYPE_t] Q):
    npts = len(Q)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef unsigned int i
    for i in range(npts):
        out[i]=1.
    return out
#################################################
@cython.boundscheck(False)
def efficiency_Ar(np.ndarray[DTYPE_t] Q):
    npts = len(Q)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef unsigned int i
    for i in range(npts):
        out[i]=1.
    return out


#################################################
@cython.boundscheck(False)
def efficiency_KIMS(np.ndarray[DTYPE_t] Q):
    npts = len(Q)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef unsigned int i
    for i in range(npts):
        out[i]=0.4
    return out

#################################################
@cython.boundscheck(False)
def efficiency_unit(np.ndarray[DTYPE_t] Q):
    npts = len(Q)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef unsigned int i
    for i in range(npts):
        out[i]=1.
    return out
