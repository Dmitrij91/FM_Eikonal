from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name='Fast_Marching_Library',
        sources=['Fast_Marching_Library.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='Distance_Utilities_Cython',
        sources=['Distance_Utilities_Cython.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='Fast_Marching_Graph_Utilities',
        sources=['Fast_Marching_Graph_Utilities.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='Fast_Marching_Non_Local_Tools',
        sources=['Fast_Marching_Non_Local_Tools.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='Fast_Marching_Binary_Heap',
        sources=['Fast_Marching_Binary_Heap.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='Fast_Marching_Energy',
        sources=['Fast_Marching_Energy.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
	include_dirs=[numpy.get_include()]),	
]

setup(
      name = 'Optimized methods',
      ext_modules = cythonize(extensions,
    compiler_directives={'language_level' : "3"})
)
