from __future__ import print_function

#from array import array
import array
import PyClConvolve

net = PyClConvolve.NeuralNet(1,28)
PyClConvolve.NetdefToNet.createNetFromNetdef( net, "rt2-8c5-mp2-16c5-mp3-150n-10n" ) 
print( net.asString() )
 
mnistFilePath = '../ClConvolve/data/mnist/t10k-dat.mat'
dim = PyClConvolve.GenericLoader.getDimensions(mnistFilePath)
print( dim )

images = array.array( 'B', [0] * (dim[0]*dim[1]*dim[2]*dim[2]) )
labels = array.array('i',[0] * dim[0] )
PyClConvolve.GenericLoader.load(mnistFilePath, images, labels, 0, 10 )

print( len(images) )
print( len(labels) )

for i in range(10):
    print( labels[i] )



