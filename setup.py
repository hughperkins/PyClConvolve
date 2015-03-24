from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("PyClConvolve",
              sources=["PyClConvolve.pyx"],
              libraries=["ClConvolve"],
              language="c++"
    )
]

setup(
  name = 'PyClConvolve',
  ext_modules = cythonize(ext_modules),
)

