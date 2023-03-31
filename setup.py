from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# python setup.py build
# python setup.py install

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# python setup.py build
# python setup.py install

src = "HyperGsys/source/"

setup(
    name='HyperGsys',
    version="0.1",
    author="fishming ys henry genghan",
    description="HyperGsys",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "dgNN"},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    ext_modules=[
        CUDAExtension(
            'hgnnaggr', [src+'hgnnaggr/hgnnaggr.cc', src+'hgnnaggr/hgnnaggr_kernel.cu']),
        CUDAExtension(
            'unignnaggr', [src+'unignnaggr/unignnaggr.cc', src+'unignnaggr/unignnaggr_kernel.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch']
)
