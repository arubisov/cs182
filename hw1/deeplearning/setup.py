from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension('im2col_cython', ['im2col_cython.pyx'],
              include_dirs=[numpy.get_include()]
              ),
]

setup(
    ext_modules=cythonize(extensions, 
        language_level = "2" # Added by Anton for compatibility to Cython=0.29
        ),
)
