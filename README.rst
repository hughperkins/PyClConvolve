PyClConvolve
============

Python wrapper for ClConvolve

How to use
==========

See `test\_clconvolve.py <test_clconvolve.py>`__ for an example of: \*
creating a network, with several layers \* loading mnist data \*
training the network using a higher-level interface (``NetLearner``)

For examples of using lower-level entrypoints, see
`test\_lowlevel.py <test_lowlevel.py>`__: \* creating layers directly \*
running epochs and forward/backprop directly

Notes on how the wrapper works
==============================

-  `cClConvolve.pxd <cClConvolve.pxd>`__ contains the definitions of the
   underlying ClConvolve c++ libraries classes
-  `PyClConvolve.pyx <PyClConvolve.pyx>`__ contains Cython wrapper
   classes around the underlying c++ classes
-  `setup.py <setup.py>`__ is a setup file for compiling the
   ``PyClConvolve.pyx`` Cython file

To build
========

Should probably more or less build on Windows too, but here are
instructions for linux for now: \* checkout:

::

    git clone --recursive https://github.com/hughperkins/PyClConvolve.git

-  build the C++ library:

   ::

       cd ClConvolve
       mkdir build
       cd build
       cmake ..
       make -j 4
       cd ../..

-  build the Cython modules

   ::

       CFLAGS="-IClConvolve/src -IClConvolve/OpenCLHelper -std=c++11" LDFLAGS="-LClConvolve/build" python setup.py build_ext -i 2>&1 | less

-  run one of the example scripts:

   ::

       LD_LIBRARY_PATH=ClConvolve/build python test_clconvolve.py
       LD_LIBRARY_PATH=ClConvolve/build python test_lowlevel.py


