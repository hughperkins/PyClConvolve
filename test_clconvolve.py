import PyClConvolve

net = PyClConvolve.NeuralNet(3,10)
PyClConvolve.NetdefToNet.createNetFromNetdef( net, "32c5-10n" ) 
print net.asString()

