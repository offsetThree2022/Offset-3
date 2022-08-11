import os
import sys
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[src for src in sources]
    )
    return cuda_ext


class get_pybind_include(object):
    """
        Helper class to deteermine the pybind11 include path
        The purpose of this class is to postpone importing pybind11
        until it is actually installed, so that the ``get_include()``
        method can be invoked
    """
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    CUDAExtension(
        name='Offset_cuda',
        sources=[
            'src/Offset2.cpp',
            'src/Offset2_kernel.cu',
        ]
    ),
]

setup(
    name='cpplib',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)

