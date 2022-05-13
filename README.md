# dside
Design space identification tool for plotting and analysing design spaces (2D and 3D). Constraints with respect to key performance indicators (KPIs) are used to categorize the samples. Convex hull algorithm (alpha shape on MATLAB) is used to identify normal operating region (NOR) and quantifying the size of the region. Given nominal point, a flexible region can be quantified to find the maximum uniform allowable disturbance that the process can still handle while satisfying all constraints.


## Installation
Currently, dside requires pandas, numpy, matplotlib, and matlab engine (for alphashapes). dside can be installd with the following commands.
```bash
pip install dside
```
For more information on how to install matlab engine please checkout this link: https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html.

## Quick Overview
Use this tool to visualize 2D and 3D design spaces, calculate NOR, and, UPAR.

```
import dside
ds = dside.DSI(df)         # Create instance of design space ds with data from DataFrame df
p = ds.screen(constraints) # Screen the points using the constraints (dictionary)
r = ds.plot(vnames)        # Plot the design space and NOR based on vnames (list of variable names for the axes)
r = ds.flex_space(x)       # Plot the nominal point and flexibility region based on point x (list/numpy array)
```

![image](https://github.com/stvsach/dside/blob/master/Fig/2D.jpg)
![image](https://github.com/stvsach/dside/blob/master/Fig/3D.jpg)