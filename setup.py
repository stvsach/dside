from setuptools import find_packages, setup

__version__ = '0.1.6.4'
short_description = 'Design space identification tool for plotting and analysing design'+\
    ' spaces (2D and 3D).'
with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name = 'dside',
    version = __version__,
    author = 'Steven Sachio',
    author_email = 'stevensachio1506@gmail.com',
    description = short_description,
    long_description = long_description,
    long_description_content_type="text/markdown",
    license = 'MIT',
    url = 'https://github.com/stvsach/dside',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Education',
    ],                                       # Information to filter the project on PyPi
    packages = find_packages(where = 'src'), # List of all python modules to be installed
    package_dir = {'': 'src'},
    python_requires='>=3.9',                 # Minimum version requirement of the package
    py_modules=['dside'],                    # Name of the python package
    install_requires=['numpy',
                      'matplotlib',
                      'pandas']              # Install other dependencies if any
)