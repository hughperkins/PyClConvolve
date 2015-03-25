from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "NeuralNet.h":
    cdef cppclass NeuralNet:
        #pass
        #def NeuralNet()
        #void print()
        NeuralNet( int numPlanes, int size )
        string asString()
    
cdef extern from "NetdefToNet.h":
    cdef cppclass NetdefToNet:
        @staticmethod
        bool createNetFromNetdef( NeuralNet *net, string netdef )

#cdef extern from "NetLearner.h":

cdef extern from "GenericLoader.h":
    cdef cppclass GenericLoader:
        @staticmethod
        void getDimensions( string trainFilepath, int *p_numExamples, int *p_numPlanes, int *p_imageSize )
        @staticmethod
        void load( string trainFilepath, unsigned char *images, int *labels, int startN, int numExamples )

#cdef extern from "mycpp.h":
#    void sayName( string name )
#    cdef cppclass MyClass:
#        # MyClass();
 #        string warpName( string inName )


