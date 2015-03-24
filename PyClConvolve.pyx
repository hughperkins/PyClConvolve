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

