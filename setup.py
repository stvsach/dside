from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name = 'dside',
    version = '0.1.0',
    author = 'Steven Sachio',
    author_email = 'stevensachio1506@gmail.com',
    description = 'Design space identification tools using pandas dataframe and MATLAB engine',
    long_description = long_description,
    license = 'MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Education',
    ],                                               # Information to filter the project on PyPi website
    packages = find_packages(),                      # List of all python modules to be installed
    python_requires='>=3.9',                         # Minimum version requirement of the package
    py_modules=['dside'],                            # Name of the python package
    install_requires=[]                              # Install other dependencies if any
)