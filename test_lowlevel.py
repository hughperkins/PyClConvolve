from __future__ import print_function

#from array import array
import array
import PyClConvolve

net = PyClConvolve.NeuralNet(1,28)
net.addLayer( PyClConvolve.NormalizationLayerMaker().translate(-0.5).scale(1/255.0) )
net.addLayer( PyClConvolve.FullyConnectedMaker().numPlanes(150).imageSize(1).biased().tanh() )
net.addLayer( PyClConvolve.FullyConnectedMaker().numPlanes(10).imageSize(1).biased().linear() )
print( net.asString() )

