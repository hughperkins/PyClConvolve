# Copyright Hugh Perkins 2015
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

from cython cimport view
from cpython cimport array as c_array
from array import array
cimport cClConvolve

cdef class NeuralNet:
    cdef cClConvolve.NeuralNet *thisptr

    def __cinit__(self, planes, size):
        self.thisptr = new cClConvolve.NeuralNet(planes, size)

    def asString(self):
        return self.thisptr.asString()

#    def myprint(self):
#        self.thisptr.print()

    def setBatchSize( self, int batchSize ):
        self.thisptr.setBatchSize( batchSize ) 
    #def propagate( self, const unsigned char[:] images):
    #    self.thisptr.propagate( &images[0] )
    def propagate( self, const float[:] images):
        self.thisptr.propagate( &images[0] )
    def backPropFromLabels( self, float learningRate, int[:] labels):
        return self.thisptr.backPropFromLabels( learningRate, &labels[0] ) 
    def backProp( self, float learningRate, float[:] expectedResults):
        return self.thisptr.backProp( learningRate, &expectedResults[0] )
    def calcNumRight( self, int[:] labels ):
        return self.thisptr.calcNumRight( &labels[0] )
    def addLayer( self, LayerMaker2 layerMaker ):
        self.thisptr.addLayer( layerMaker.baseptr )

cdef class NetdefToNet:
    @staticmethod
    def createNetFromNetdef( NeuralNet neuralnet, netdef ):
        return cClConvolve.NetdefToNet.createNetFromNetdef( neuralnet.thisptr, netdef )

cdef class NetLearner: 
    cdef cClConvolve.NetLearner[float] *thisptr
    def __cinit__( self, NeuralNet neuralnet ):
        self.thisptr = new cClConvolve.NetLearner[float]( neuralnet.thisptr )
    def setTrainingData( self, Ntrain, float[:] trainData, int[:] trainLabels ):
        self.thisptr.setTrainingData( Ntrain, &trainData[0], &trainLabels[0] )
    def setTestingData( self, Ntest, float[:] testData, int[:] testLabels ):
        self.thisptr.setTestingData( Ntest, &testData[0], &testLabels[0] )
    def setSchedule( self, numEpochs ):
        self.thisptr.setSchedule( numEpochs )
    def setDumpTimings( self, bint dumpTimings ):
        self.thisptr.setDumpTimings( dumpTimings )
    def setBatchSize( self, batchSize ):
        self.thisptr.setBatchSize( batchSize )
    def learn( self, learningRate ):
        self.thisptr.learn( learningRate )

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
    def loaduc( trainFilepath, unsigned char[:] images, int[:] labels, startN, numExamples ):
        #(N, planes, size) = getDimensions(trainFilepath)
        #images = view.array(shape=(N,planes,size,size),itemsize=1,
        #cdef unsigned char *images
        #cdef int *labels
        cClConvolve.GenericLoader.load( trainFilepath, &images[0], &labels[0], startN , numExamples )
        #return (images, labels)
    @staticmethod 
    def load( trainFilepath, float[:] images, int[:] labels, startN, numExamples ):
        (N, planes, size) = GenericLoader.getDimensions(trainFilepath)
        #images = view.array(shape=(N,planes,size,size),itemsize=1,
        #cdef unsigned char *images
        #cdef int *labels
        #cdef unsigned char ucImages[numExamples * planes * size * size]
        print( (N, planes, size ) )
        cdef c_array.array ucImages = array('B', [0] * (numExamples * planes * size * size) )
        cdef unsigned char[:] ucImagesMv = ucImages
        cClConvolve.GenericLoader.load( trainFilepath, &ucImagesMv[0], &labels[0], startN , numExamples )
        #return (images, labels)
        cdef int i
        cdef int total
        total = numExamples * planes * size * size
        print(total)
        for i in range(total):
            images[i] = ucImagesMv[i]

cdef class LayerMaker2:
    cdef cClConvolve.LayerMaker2 *baseptr

cdef class NormalizationLayerMaker(LayerMaker2):
    cdef cClConvolve.NormalizationLayerMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cClConvolve.NormalizationLayerMaker()
        self.baseptr = self.thisptr
    def translate( self, float _translate ):
        self.thisptr.translate( _translate )
        return self
    def scale( self, float _scale ):
        self.thisptr.scale( _scale )
        return self
    @staticmethod
    def instance():
        return NormalizationLayerMaker()

cdef class FullyConnectedMaker(LayerMaker2):
    cdef cClConvolve.FullyConnectedMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cClConvolve.FullyConnectedMaker()
        self.baseptr = self.thisptr
    def numPlanes( self, int _numPlanes ):
        self.thisptr.numPlanes( _numPlanes )
        return self
    def imageSize( self, int _imageSize ):
        self.thisptr.imageSize( _imageSize )
        return self
    def biased(self):
        self.thisptr.biased()
        return self
    def biased(self, int _biased):
        self.thisptr.biased( _biased )
        return self
    def linear(self):
        self.thisptr.linear()
        return self
    def tanh(self):
        self.thisptr.tanh()
        return self
    def sigmoid(self):
        self.thisptr.sigmoid()
        return self
    def relu(self):
        self.thisptr.relu()
        return self
    @staticmethod
    def instance():
        return FullyConnectedMaker()


