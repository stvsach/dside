# dside
Design space identification tool for plotting and analysing design spaces (2D and 3D). Constraints with respect to key performance indicators (KPIs) are used to categorize the samples. Convex hull algorithm (alpha shape on MATLAB) is used to identify normal operating region (NOR) and quantifying the size of the region. Given nominal point, a flexible region can be quantified to find the maximum uniform allowable disturbance that the process can still handle while satisfying all constraints.


## Installation
Currently, dside requires pandas, numpy, matplotlib, and matlab engine. dside can be installd with the following commands.
```bash
pip install dside
```
For more information on how to install matlab engine please checkout this link: https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html.

## Quick Overview
Use this tool to visualize 2D and 3D design spaces, calculate NOR, and, UPAR.

```
import dside
# 1. Create instance of ds with data from DataFrame df
ds = dside.DSI(df)
# 2. Screen the points using the constraints (dictionary)
p = ds.screen(constraints)
# 3. Find DSp boundaries based on vnames (list of variable names for the axes)
shp = ds.find_DSp(vnames)
# 4. Plot the design space and the samples
r = ds.plot(vnames)
# 5. Plot the nominal point and AOR based on point x (list/numpy array)
r = ds.find_AOR(x)
# 6. Save the results in detailed .txt file
ds.send_output('output.txt')
```

![image](https://raw.githubusercontent.com/stvsach/dside/df8f03256e3913f1eb020003bfcc23cbde7e1b1c/Fig/2D.svg)
![image](https://raw.githubusercontent.com/stvsach/dside/df8f03256e3913f1eb020003bfcc23cbde7e1b1c/Fig/3D.svg)