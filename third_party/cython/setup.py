from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'Connectivity',
  ext_modules = cythonize("connectivity.pyx"),
  include_dirs=[numpy.get_include()]
)
