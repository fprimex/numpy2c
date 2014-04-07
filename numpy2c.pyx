cimport numpy
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

# Credit to Tiago Pereira for the conversion routines
# http://web.archiveorange.com/archive/v/JPeC5e4On3uR5BZQI9U2
cdef inline double **numpy2c_dbl(numpy.ndarray[numpy.float64_t, ndim=2] a):
    """Convert 2D numpy array to double** for processing in C"""
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    cdef int i
    cdef double **data
    data = <double **> malloc(m*sizeof(double *))
    for i in range(m):
        data[i] = &(<double *>a.data)[i*n]
    return data

cdef inline numpy.ndarray c2numpy_dbl(double **a, int n, int m):
    """Convert double** from C into an initialized 2D numpy array"""
    cdef numpy.ndarray[numpy.float64_t, ndim=2]result = numpy.zeros((m,n),dtype=numpy.float64)
    cdef double *dest
    cdef int i
    dest = <double *> malloc(m*n*sizeof(double*))  
    for i in range(m):
        memcpy(dest + i*n,a[i],m*sizeof(double*))
        free(a[i])
    memcpy(result.data,dest,m*n*sizeof(double*))
    free(dest)
    free(a)
    return result

