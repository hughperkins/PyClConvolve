# PyClConvolve
Python wrapper for ClConvolve

# How to use

See [test_clconvolve.py](test_clconvolve.py) for an example of:
* creating a network, with several layers
* loading mnist data
* training the network using low-level net interface (`propagate`, `backprop`)
* training the network using a higher-level interface (`NetLearner`)

# Notes on how the wrapper works

* [cClConvolve.pxd](cClConvolve.pxd) contains the definitions of the underlying ClConvolve c++ libraries classes
* [PyClConvolve.pyx](PyClConvolve.pyx) contains Cython wrapper classes around the underlying c++ classes
* [setup.py](setup.py) is a setup file for compiling the `PyClConvolve.pyx` Cython file

