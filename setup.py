#run this from terminal as: python setup.py install --user
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os, sys

def readme():
    with open('README.rst') as f:
        return f.read()

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__DMDD_SETUP__ = True
import dmdd
version = dmdd.__version__

# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

# Push a new tag to GitHub.
if "tag" in sys.argv:
    os.system("git tag -a {0} -m 'version {0}'".format(version))
    os.system("git push --tags")
    sys.exit()


    
ext_modules = [
               Extension('dmdd_efficiencies',['dmdd/dmdd_efficiencies.pyx'],include_dirs=[numpy.get_include()]),
               Extension('helpers',['dmdd/helpers.pyx'],include_dirs=[numpy.get_include()]),
               Extension('formUV',['dmdd/formUV.pyx'],include_dirs=[numpy.get_include()]),
               Extension('formNR',['dmdd/formNR.pyx'],include_dirs=[numpy.get_include()]),
               Extension('formgenNR',['dmdd/formgenNR.pyx'],include_dirs=[numpy.get_include()]),
               Extension('rate_NR',['dmdd/rate_NR.pyx'],include_dirs=[numpy.get_include()]),
               Extension('rate_genNR',['dmdd/rate_genNR.pyx'],include_dirs=[numpy.get_include()]),
               Extension('rate_UV',['dmdd/rate_UV.pyx'],include_dirs=[numpy.get_include()])
               ]

setup(name = "dmdd",
      version = version,
      description = "Enables simple simulation and Bayesian posterior analysis of recoil-event data from dark-matter direct-detection experiments under a wide variety of scattering theories. ",
      long_description = readme(),
      author = "V. Gluscevic, S. D. McDermott",
      author_email = "verag@ias.edu",
      url = "https://github.com/veragluscevic/dmdd_2014",
      packages = find_packages(),
      ext_modules = ext_modules,
      cmdclass = {'build_ext': build_ext},
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics'
        ],
      install_requires=['cython'],
      zip_safe=False
)
