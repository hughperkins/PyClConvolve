from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "NeuralNet.h":
    cdef cppclass NeuralNet:
        pass
        #def NeuralNet()
    
cdef extern from "NetdefToNet.h":
    cdef cppclass NetdefToNet:
        bool createNetFromNetdef( NeuralNet *net, string netdef )

#cdef extern from "NetLearner.h":


#cdef extern from "mycpp.h":
#    void sayName( string name )
#    cdef cppclass MyClass:
#        # MyClass();
 #        string warpName( string inName )


