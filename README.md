# dside

[![Python package](https://github.com/stvsach/dside/actions/workflows/python-package.yml/badge.svg)](https://github.com/stvsach/dside/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/dside/badge/?version=latest)](https://dside.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/dside.svg)](https://pypi.org/project/dside)
[![Downloads](https://pepy.tech/badge/dside)](https://pepy.tech/project/dside)

Design space identification tool for plotting and analysing design spaces (2D and 3D). Constraints with respect to key performance indicators (KPIs) are used to categorize the samples. Concave hulls (alpha shape) are used to identify design space (DSp) and quantify the size of the space. Given nominal operating point (NOP), an acceptable operating region (AOR) can be quantified to find the maximum multivariate allowable disturbance that the process can still handle while satisfying all constraints (multivariate proven acceptable range - MPAR).


## Installation
Currently, dside requires pandas, numpy, matplotlib, and shapely. dside can be installd with the following commands.
```bash
pip install dside
```

## Quick Overview
Use this tool to visualize 2D and 3D design spaces, calculate NOR, and, MPAR.

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
# 6. Save the results in detailed output.txt file
ds.send_output('output')
```

![image](https://github.com/stvsach/dside/blob/main/Fig/2D.svg)
![image](https://github.com/stvsach/dside/blob/main/Fig/3D.svg)