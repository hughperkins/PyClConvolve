from __future__ import print_function

#from array import array
import array
import PyClConvolve

net = PyClConvolve.NeuralNet()
net.addLayer( PyClConvolve.InputLayerMaker().numPlanes(1).imageSize(28) )
net.addLayer( PyClConvolve.NormalizationLayerMaker().translate(-0.5).scale(1/255.0) )
net.addLayer( PyClConvolve.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased().relu() )
net.addLayer( PyClConvolve.PoolingMaker().poolingSize(2) )
net.addLayer( PyClConvolve.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased().relu() )
net.addLayer( PyClConvolve.PoolingMaker().poolingSize(3) )
net.addLayer( PyClConvolve.FullyConnectedMaker().numPlanes(150).imageSize(1).biased().tanh() )
net.addLayer( PyClConvolve.FullyConnectedMaker().numPlanes(10).imageSize(1).biased().linear() )
net.addLayer( PyClConvolve.SquareLossMaker() )
print( net.asString() )

