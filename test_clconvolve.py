import PyClConvolve

net = PyClConvolve.NeuralNet(1,28)
PyClConvolve.NetdefToNet.createNetFromNetdef( net, "rt2-8c5-mp2-16c5-mp3-150n-10n" ) 
print net.asString()

