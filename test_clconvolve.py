from __future__ import print_function

#from array import array
import array
import PyClConvolve

net = PyClConvolve.NeuralNet(1,28)
PyClConvolve.NetdefToNet.createNetFromNetdef( net, "rt2-8c5-mp2-16c5-mp3-150n-10n" ) 
print( net.asString() )
 
mnistFilePath = '../ClConvolve/data/mnist/t10k-dat.mat'
(N,planes,size) = PyClConvolve.GenericLoader.getDimensions(mnistFilePath)
print( (N,planes,size) )

N = 1280
images = array.array( 'f', [0] * (N*planes*size*size) )
labels = array.array('i',[0] * N )
PyClConvolve.GenericLoader.load(mnistFilePath, images, labels, 0, N )

for i in range(N * planes * size * size):
    images[i] = images[i] / 255.0 - 0.5

net.setBatchSize(128)
for i in range(12): 
    net.propagate( images )
    net.backPropFromLabels( 0.002, labels )
    print( 'numright ' + str( net.calcNumRight( labels ) ) )
#    print( 'loss ' + str( loss ) )
 
netLearner = PyClConvolve.NetLearner( net )
netLearner.setTrainingData( N, images, labels )
netLearner.setTestingData( N, images, labels )
netLearner.setSchedule( 12 )
netLearner.setBatchSize( 128 )
netLearner.learn( 0.002 )
 

