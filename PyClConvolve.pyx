cimport cClConvolve
  
cdef class PyNeuralNet:
    cdef cClConvolve.NeuralNet *thisptr

    def __cinit__(self):
        self.thisptr = new cClConvolve.NeuralNet()



