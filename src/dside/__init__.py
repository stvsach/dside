from dside.ds3D import*

try:
    import matlab.engine
except ModuleNotFoundError:
    print('WARNING: matlab engine not found. Alpha shape of design space cannot be calculated.\nInstall from: https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html')
