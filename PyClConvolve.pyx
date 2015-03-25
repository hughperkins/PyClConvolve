from cython cimport view
from cpython cimport array as c_array
cimport cClConvolve

cdef class NeuralNet:
    cdef cClConvolve.NeuralNet *thisptr

    def __cinit__(self, planes, size):
        self.thisptr = new cClConvolve.NeuralNet(planes, size)

    def asString(self):
        return self.thisptr.asString()

#    def myprint(self):
#        self.thisptr.print()

cdef class NetdefToNet:
    @staticmethod
    def createNetFromNetdef( NeuralNet neuralnet, netdef ):
        return cClConvolve.NetdefToNet.createNetFromNetdef( neuralnet.thisptr, netdef )

cdef class GenericLoader:
    @staticmethod
    def getDimensions( trainFilePath ):
        cdef int N
        cdef int planes
        cdef int size
        cClConvolve.GenericLoader.getDimensions( trainFilePath, &N, &planes, &size )
        # print( N )
        return (N,planes,size)
    @staticmethod
    def load( trainFilepath, unsigned char[:] images, int[:] labels, startN, numExamples ):
        #(N, planes, size) = getDimensions(trainFilepath)
        #images = view.array(shape=(N,planes,size,size),itemsize=1,
        #cdef unsigned char *images
        #cdef int *labels
        cClConvolve.GenericLoader.load( trainFilepath, &images[0], &labels[0], startN , numExamples )
        #return (images, labels)

