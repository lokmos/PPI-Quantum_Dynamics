from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys
import os

# 判断平台
is_windows = sys.platform.startswith("win")

# 设置编译参数
if is_windows:
    extra_compile_args = []
    extra_link_args = []
else:
    extra_compile_args = ["-ffast-math", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

extensions = [
    Extension(
        name="dyn_cython",
        sources=["dyn_cython.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )
]

setup(
    name="dyn_cython",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
