#from distutils.core import setup
import os
import os.path
import sysconfig
import sys
from setuptools import setup
#from distutils.extension import Extension
from setuptools import Extension
try:
    from Cython.Build import cythonize
    cython_present = True
except ImportError:
    pass
try:
    import pypandoc
    pypandoc_present = True
except ImportError:
    pass

print ( sys.argv )

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def get_so_suffix():
    if sysconfig.get_config_var('SOABI') != None:
        return "." + sysconfig.get_config_var('SOABI')
    return ""

def read_md( mdname ): 
    if pypandoc_present:
        return pypandoc.convert(mdname, 'rst')
    else:
        print("warning: pypandoc module not found, could not convert Markdown to RST")
        return open(mdname, 'r').read()

if pypandoc_present:
    pypandoc.convert('README.md', 'rst', outputfile = 'README.rst' )

def my_cythonize(extensions, **_ignore):
    #newextensions = []
    for extension in extensions:
        print(extension.sources)
        should_cythonize = False
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                should_cythonize = True
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        print(should_cythonize)
        if should_cythonize and cython_present:
            cythonize(extension)
        extension.sources[:] = sources    
        #newextensions.append( extension )
    return extensions

# from http://stackoverflow.com/questions/14320220/testing-python-c-libraries-get-build-path
def distutils_dir_name(dname):
    """Returns the name of a distutils build directory"""
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(dirname=dname,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info)
 
def lib_build_dir():
    return os.path.join('build', distutils_dir_name('lib'))

clconvolve_sourcestring = """LayerMaker.cpp NeuralNetMould.cpp
     ConvolutionalLayer.cpp NeuralNet.cpp Layer.cpp InputLayer.cpp
    Propagate1.cpp Propagate.cpp Propagate2.cpp Propagate3.cpp LayerDimensions.cpp
    Propagate4.cpp ActivationFunction.cpp SquareLossLayer.cpp LossLayer.cpp BackpropWeights2.cpp
    BackpropWeights2Cpu.cpp BackpropErrorsv2.cpp BackpropErrorsv2Cpu.cpp
    BackpropWeights2Naive.cpp BackpropErrorsv2Naive.cpp BackpropWeights2Scratch.cpp
    CrossEntropyLoss.cpp SoftMaxLayer.cpp FullyConnectedLayer.cpp  EpochMaker.cpp
    PoolingPropagate.cpp PoolingPropagateCpu.cpp PoolingLayer.cpp PoolingBackprop.cpp
    PoolingBackpropCpu.cpp PoolingPropagateGpuNaive.cpp BackpropWeights2ScratchLarge.cpp
    BatchLearner.cpp NetdefToNet.cpp NetLearner.cpp stringhelper.cpp NormalizationLayer.cpp
    RandomPatches.cpp RandomTranslations.cpp NorbLoader.cpp MultiNet.cpp
    Trainable.cpp InputLayerMaker.cpp ConvolutionalMaker.cpp RandomTranslationsMaker.cpp
    RandomPatchesMaker.cpp NormalizationLayerMaker.cpp FullyConnectedMaker.cpp
    PoolingMaker.cpp PatchExtractor.cpp Translator.cpp GenericLoader.cpp Kgsv2Loader.cpp
    BatchLearnerOnDemand.cpp NetLearnerOnDemand.cpp BatchProcess.cpp WeightsPersister.cpp
    PropagateFc.cpp BackpropErrorsv2Cached.cpp PropagateByInputPlane.cpp
    PropagateExperimental.cpp PropagateAuto.cpp PropagateCpu.cpp Propagate3_unfactorized.cpp
    PoolingBackpropGpuNaive.cpp""" 
clconvolve_sources_all = clconvolve_sourcestring.split()
clconvolve_sources = []
for source in clconvolve_sources_all:
    clconvolve_sources.append(source)

openclhelpersources = list(map( lambda name : 'ClConvolve/' + name, [ 'OpenCLHelper/OpenCLHelper.cpp',
        'OpenCLHelper/deviceinfo_helper.cpp', 'OpenCLHelper/platforminfo_helper.cpp',
        'OpenCLHelper/CLKernel.cpp', 'OpenCLHelper/thirdparty/clew/src/clew.c' ] ))
print(openclhelpersources)
print(isinstance( openclhelpersources, list) )

ext_modules = [
    Extension("libOpenCLHelper",
        sources = openclhelpersources,
        include_dirs = ['ClConvolve/OpenCLHelper'],
        extra_compile_args=['-std=c++11']
#        libraries = []
#        language='c++'
    ),
    Extension("libClConvolve",
        list(map( lambda name : 'ClConvolve/src/' + name, clconvolve_sources )),
        include_dirs = ['ClConvolve/src','ClConvolve/OpenCLHelper'],
        extra_compile_args = ['-std=c++11'],
        library_dirs = [ lib_build_dir() ],
        libraries = [ "OpenCLHelper" + get_so_suffix() ],
#        language='c++'
    ),
    Extension("PyClConvolve",
              sources=["PyClConvolve.pyx"],
              include_dirs = ['ClConvolve/src','ClConvolve/OpenCLHelper'],
              libraries=["ClConvolve" + get_so_suffix() ],
              extra_compile_args=['-std=c++11'],
              library_dirs = [lib_build_dir()],
              language="c++"
    )
]

setup(
  name = 'PyClConvolve',
  version = "0.0.2",
  author = "Hugh Perkins",
  author_email = "hughperkins@gmail.com",
  description = 'python wrapper for ClConvolve deep convolutional neural network library for OpenCL',
  license = 'MPL',
  url = 'https://github.com/hughperkins/PyClConvolve',
  long_description = read_md('README.md'),
  classifiers = [
    'Development Status :: 4 - Beta',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
  ],
  install_requires = ['Cython>=0.22','cogapp>=2.4','future>=0.14.3'],
  tests_require = ['nose>=1.3.4'],
 # modules = libraries,
  ext_modules = my_cythonize( ext_modules),
)

