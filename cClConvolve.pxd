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
        #def NeuralNet()
        #void print()
        NeuralNet( int numPlanes, int size )
        string asString()
        void setBatchSize( int batchSize )
        void propagate( const float *images)
        void backPropFromLabels( float learningRate, const int *labels)
        void backProp( float learningRate, const float *expectedResults)
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
        void learn( float learningRate )
        #void setSchedule( int numEpochs, int startEpoch )
        # VIRTUAL void addPostEpochAction( PostEpochAction *action );
        #void learn( float learningRate, float annealLearningRate )

cdef extern from "GenericLoader.h":
    cdef cppclass GenericLoader:
        @staticmethod
        void getDimensions( string trainFilepath, int *p_numExamples, int *p_numPlanes, int *p_imageSize )
        @staticmethod
        void load( string trainFilepath, unsigned char *images, int *labels, int startN, int numExamples )

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
        FullyConnectedMaker *numPlanes( float numPlanes )
        FullyConnectedMaker *imageSize( float imageSize )
        FullyConnectedMaker *biased()
        FullyConnectedMaker *biased(int _biased)
        FullyConnectedMaker *linear()
        FullyConnectedMaker *tanh()
        FullyConnectedMaker *sigmoid()
        FullyConnectedMaker *relu()
        @staticmethod
        FullyConnectedMaker *instance()



