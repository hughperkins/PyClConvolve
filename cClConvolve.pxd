# Copyright Hugh Perkins 2015
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "NeuralNet.h":
    cdef cppclass NeuralNet:
        #pass
        NeuralNet()
        #void print()
        NeuralNet( int numPlanes, int size ) except +
        string asString() except +
        void setBatchSize( int batchSize ) except +
        void propagate( const float *images) except +
        void backPropFromLabels( float learningRate, const int *labels) except +
        void backProp( float learningRate, const float *expectedResults) except +
        int calcNumRight( const int *labels )
        void addLayer( LayerMaker2 *maker )

cdef extern from "NetdefToNet.h":
    cdef cppclass NetdefToNet:
        @staticmethod
        bool createNetFromNetdef( NeuralNet *net, string netdef )

cdef extern from "NetLearner.h":
    cdef cppclass NetLearner[T]:
        NetLearner( NeuralNet *net )
        void setTrainingData( int Ntrain, T *trainData, int *trainLabels )
        void setTestingData( int Ntest, T *testData, int *testLabels )
        void setSchedule( int numEpochs )
        void setDumpTimings( bool dumpTimings )
        void setBatchSize( int batchSize )
        void learn( float learningRate ) except +
        #void setSchedule( int numEpochs, int startEpoch )
        # VIRTUAL void addPostEpochAction( PostEpochAction *action );
        #void learn( float learningRate, float annealLearningRate )

cdef extern from "GenericLoader.h":
    cdef cppclass GenericLoader:
        @staticmethod
        void getDimensions( string trainFilepath, int *p_numExamples, int *p_numPlanes, int *p_imageSize ) except +
        @staticmethod
        void load( string trainFilepath, unsigned char *images, int *labels, int startN, int numExamples ) except +

cdef extern from "LayerMaker.h":
    cdef cppclass LayerMaker2:
        pass

cdef extern from "NormalizationLayerMaker.h":
    cdef cppclass NormalizationLayerMaker(LayerMaker2):
        NormalizationLayerMaker *translate( float translate )
        NormalizationLayerMaker *scale( float scale )
        @staticmethod
        NormalizationLayerMaker *instance()

cdef extern from "FullyConnectedMaker.h":
    cdef cppclass FullyConnectedMaker(LayerMaker2):
        FullyConnectedMaker *numPlanes( int numPlanes )
        FullyConnectedMaker *imageSize( int imageSize )
        FullyConnectedMaker *biased()
        FullyConnectedMaker *biased(bint _biased)
        FullyConnectedMaker *linear()
        FullyConnectedMaker *tanh()
        FullyConnectedMaker *sigmoid()
        FullyConnectedMaker *relu()
        @staticmethod
        FullyConnectedMaker *instance()

cdef extern from "ConvolutionalMaker.h":
    cdef cppclass ConvolutionalMaker(LayerMaker2):
        ConvolutionalMaker *numFilters( int numFilters )
        ConvolutionalMaker *filterSize( int imageSize )
        ConvolutionalMaker *padZeros()
        ConvolutionalMaker *padZeros(bint _padZeros)
        ConvolutionalMaker *biased()
        ConvolutionalMaker *biased(bint _biased)
        ConvolutionalMaker *linear()
        ConvolutionalMaker *tanh()
        ConvolutionalMaker *sigmoid()
        ConvolutionalMaker *relu()
        @staticmethod
        ConvolutionalMaker *instance()

cdef extern from "PoolingMaker.h":
    cdef cppclass PoolingMaker(LayerMaker2):
        PoolingMaker *poolingSize( int _poolingsize )
        PoolingMaker *padZeros( int _padZeros )
        @staticmethod
        PoolingMaker *instance()

cdef extern from "LayerMaker.h":
    cdef cppclass SquareLossMaker(LayerMaker2):
        @staticmethod
        SquareLossMaker *instance()

cdef extern from "LayerMaker.h":
    cdef cppclass SoftMaxMaker(LayerMaker2):
        @staticmethod
        SoftMaxMaker *instance()

cdef extern from "InputLayerMaker.h":
    cdef cppclass InputLayerMaker[T](LayerMaker2):
        InputLayerMaker *numPlanes( int _numPlanes )
        InputLayerMaker *imageSize( int _imageSize )
        @staticmethod
        InputLayerMaker *instance()

