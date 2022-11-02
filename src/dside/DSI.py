def get_at_time(df, t, time_label = 'Time'):
    """
    Slice df which is a multi index dataframe
    use this function to get the element at time t of each sample
    if t = -1 then it will be the last element
    """
    import numpy as np
    import pandas as pd
    dft = {}
    sample_index = list(df.index.levels[0])
    no_sams = len(sample_index)

    if t == -1:
        for i in range(no_sams):
            dft[i] = df.loc[i].iloc[-1]
    else:
        for i in range(no_sams):
            sliced = df.loc[i][df.loc[i][time_label].round() == np.round(t)]
            if sliced.size == 0:
                pass
            else:
                dft[i] = sliced.iloc[0]
    return pd.DataFrame(dft).T

def qp(df, constraints, vnames, x = None, opt = None):
    """
    Quick plotting function. Uses the DSI class.
    df: Pandas DataFrame containing data
    constraints: dictionary containing name of variable and list of 
                                    [lower bound, upper bound]
    vnames: list of manipulated variable names
    x: list of nominal point for acceptable operating region analysis
    opt: options for the lotting
    """
    from dside import DSI
    
    ds = DSI(df)
    p = ds.screen(constraints)
    r = ds.plot(vnames, opt)
    if x != None:
        r = ds.find_AOR(x)
    return ds

class DSI():
    """
    Design space identification framework
    Takes in df -> pandas dataframe of labelled data
    """
    def __init__(self, df, **kwargs):
        """
        Initialize internal definitions
        df: labelled pandas dataframe
        """
        # ----- Imports ----- #
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        # Set internal definition
        self.df = df
        self.report = {
            'nosam': df.shape[0], 'space_size': '-',
            'hmv': {'name': 'None', 'mean': '-', 'max': '-', 'max_sample': '-',\
                 'min': '-',  'min_sample': '-'},
        }
        self.all_x = {}
        self.space_size = None
        self.tetra = None
        self.r = None
        
        # Set default color map for heat map plot
        inferno_modified = plt.cm.get_cmap('inferno', 256)
        gray_modified    = plt.cm.get_cmap('gray', 256)
        self.inferno80   = ListedColormap(inferno_modified(np.linspace(0, 0.8, 256)))
        self.gray80      = ListedColormap(gray_modified(np.linspace(0, 0.85, 256)))
        self.cmap_opt = {'inferno80': self.inferno80, 'gray80': self.gray80}
        
        # Default options
        self.default_opt = {
            # ----- Saving Options ----- #
            'save_flag': False, # If True, then figure will be saved
            'save_name': 'test.jpg', # save file name including path
            'save_dpi': 480, # save dpi
            
            # ----- Labels ----- #
            'xlabel': 'x',       # x axis label
            'ylabel': 'y',       # y axis label
            'zlabel': 'z',       # z axis label
            'cmap': 'inferno80', # Matplotlib colour map
            'hmv': 'None',       # heat map variable name
            'hmvlabel': 'hmvlabel: heat map var label', # heat map variable label
            
            # ----- Hidden Elements ----- #
            'hidehmv': False, # If True, no heat map will be plotted
            'hidesat': False, # If True, no satisfied variables will be plotted
            'hidevio': False, # If True, no violated variables will be plotted
            'hidedsp': False, # If True, hides the surface/boundary
            
            # ----- Plot Format ----- #
            'czorder': False,     # computed_zorder for 3D plots settings
            'fs': (6, 4),         # Figure size
            'bw': False,          # If True, use black-white template
            'alpha': 1,           # Transparency of points
            'font_size': 14,      # Font size
            # Color bar and map
            'cbarloc': 'right',   # Colobar location
            'cbaror': 'vertical', # Colorbar orientation
            'cbarpad': 0.12,      # Colorbar padding
            'mycmap': 'viridis',  # Use your own cmap (input name of cmap as str)
            # Satisfied samples
            'satlabel': 'Sat',     # Satisfied samples label
            'satcolor': '#FF9000', # Satisfied samples color
            'satmarker': '.',      # Marker of satisfied points
            'satfill': '#FF9000',  # Marker fill color of satisfied points
            'satzorder': 5,        # Decides which level to be plotted on
            # Violated samples
            'violabel': 'Vio',     # Violated samples label
            'viocolor': '#005DC1', # Violated samples color
            'viomarker': '.',      # Marker of violated points
            'viofill': '#005DC1',  # Marker fill color of violated points
            'viozorder': 5,        # Decides which level to be plotted on
            # Legend
            'legloc': 'upper right', # Legend location
            'framealpha': 1,         # Legend box transparency
            'legendzorder': 100,     # Decides which level to be plotted on
            # 3D view
            'elev': 20,              # Elevation of 3D plot
            'azim': -70,             # Azimuth of 3D plot
            # Axes limit
            'limfactor': 0.05,       # Axes limit factor based on range of axes
            'axeslimdf': 'df',       # Data used to calculate axes limits 
                                     #          ('sat', 'vio', 'df', or 'best')
            
            # ----- Design Space Parameters ----- #
            'dsplabel': 'DSp',   # Label of surface/boundary
            'dspcolor': 'black', # Color of the surface/boundary (both)
            'dspwidth': 4,       # Thickness of the boundary (2D)
            'dspstyle': '-',     # Line style of the boundary (2D)
            'dspalpha': 0.2,     # Transparency of surface/boundary (3D)
            'dspzorder': 20,     # To make sure it is plotted ontop of the samples
            
            # ----- NOP Parameters ----- #
            'step_change': 1,    # Step change of expanding AOR in percent
            'noplabel': 'NOP',   # Normal Operating Point label for legend
            'nopmarker': 'x',    # Nominal operating point marker style
            'nopwidth':   4,     # Nominal operating point marker thickness
            'nopcolor': 'black', # Nominal operating point marker color
            'nopsize':  150,     # Nominal operating point marker size
            'nopzorder': 10,     # To make sure it is plotted ontop of the samples

            # ----- AOR Parameters ----- #
            'aorlabel': 'AOR',   # Uniform Proven Acceptable Range
            'aorstyle': '--',    # AOR boundary line style
            'aorcolor': 'black', # AOR boundary line color
            'aorwidth': 4,       # AOR boundary line width
            'aorzorder': 10,     # To make sure it is plotted ontop of the samples
            'AORlb': 0,          # Bisection params (lower bound)
            'AORub': 1,          # Bisection params (upper bound)
            'AORtol': 0.001,     # Bisection params - tolerance
            'AORmaxiter': 50,    # Bisection params - max iterations
            'AORprintF': False,  # Print bisection output
            
            # ----- Hull Parameters ----- #
            'no_splits': 10, # Splits for the inside3D function to reduce memory req
            'a': None,        # Alpha value -> at large alpha,
                              # hull becomes convex. if None: use product of bounds range
            'amul': 1,        # Alpha multiplier value (wrt to a used)
            'opt_amul': True, # If True, use bisection to find largest amul with no vio
                              # in DSp
            'extra_points': [], # Extra points to be used to design space identification
            'maxiter': 50,    # Maximum bisection iterations
            'tol': 0.001,     # Bisection tolerance (on the MV which is amul)
            'lb': 0.001,      # Lower bound of initial bisection run
            'ub': 5,          # Upper bound of initial bisection run
            'printF': False,  # If true, print iter details
            'maxvp': 0,       # Maximum allowed percentage of vio in DSp
        }
        self.opt = self.default_opt.copy()
        self.bw_template = {'alpha': 1, 'satcolor': 'gray', 'viocolor': 'black', \
            'satfill': 'gray', 'viofill': 'black', 'cmap': 'gray80',
                           'satmarker': 'o', 'viomarker': 'o'}
        return None
    
    def reset(self):
        """
        Reset options to default.
        """
        self.opt = self.default_opt.copy()
        return None
    
    def update_opt(self, new_opt):
        """
        Update current self.opt from contents of new_opt dictionary.
        """
        self.opt.update(new_opt)
        return None
        
    def screen(self, constraints):
        """
        Takes in the DataFrame, data, and dictionary, constraints, 
        giving out the satisfied and violated DataFrame of samples
        constraints: {'output_name1': [lbd1, ubd1], 'output_name2': [lbd1, ubd2], ...}
        """
        import pandas as pd
        data = self.df
        self.constraints = constraints
        sat = data.copy()
        for i in list(constraints.keys()):
            sat = sat[sat[i] >= constraints[i][0]]
            sat = sat[sat[i] <= constraints[i][1]]
        exclude_these = data.index.isin(list(sat.index))
        vio = data[~exclude_these]

        vSatFlag = pd.DataFrame([False for i in range(vio.shape[0])], columns = ['SatFlag'])
        vSatFlag.index = vio.index
        vio = pd.concat([vio, vSatFlag], axis = 1)
        
        sSatFlag = pd.DataFrame([False for i in range(sat.shape[0])], columns = ['SatFlag'])
        sSatFlag.index = sat.index
        sat = pd.concat([sat, sSatFlag], axis = 1)

        sat['SatFlag'] = True
        vio['SatFlag'] = False
        self.sat = sat
        self.vio = vio
        return sat, vio
    
    def help(self, print_opt = False):
        """
        Prints usage instructions and ALL of the current options and return the 
        opt dictionary.
        """
        print('# ----- Usage Instructions ----- #')
        print('# 1. Create instance of ds with data from DataFrame df')
        print('ds = dside.DSI(df)')
        print('# 2. Screen the points using the constraints (dictionary)')
        print('p = ds.screen(constraints)')
        print('# 3. Find DSp boundaries based on vnames (list of variable names for the axes)')
        print('shp = ds.find_DSp(vnames)')
        print('# 4. Plot the design space and the samples')
        print('r = ds.plot(vnames)')
        print('# 5. Plot the nominal point and AOR based on point x (list/numpy array)')
        print('r = ds.find_AOR(x)')
        print('# 6. Save the results in detailed output.txt file and output.pkl file')
        print("ds.send_output('output')")
        if print_opt:
            print('\n# ----- Options ----- #')
            for i in list(self.opt.keys()):
                print(f'{i:10}: {self.opt[i]}')
        return self.opt
    
    def plot(self, vnames = None, opt = {}):
        """
        Plot either 2D or 3D design space based on satisfied and violated points.
        vnames: ['varname1', 'varname2', 'varname3']
        Plotting options available by passing opt 
        which is a dictionary to update the default self.opt.
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        # ---------------------------- Internal definitions ---------------------------- #
        if vnames == None:
            vnames = self.vnames
        else:
            self.vnames = vnames
        axes_label = ['xlabel', 'ylabel', 'zlabel']
        for i, l in enumerate(vnames):
            self.opt[axes_label[i]] = l
        self.opt.update(opt)
        if self.opt['bw']:
            self.opt.update(self.bw_template)
        self.opt.update(opt)
        
        # ---------------------------- External definitions ---------------------------- #
        sat = self.sat
        vio = self.vio
        opt = self.opt    
        plt.rcParams.update({'font.size': opt['font_size']})
        
        # ----------------------------- Heat map Variable ----------------------------- #
        hmv_R = {'name': opt['hmv'], 'mean': '-', 'max': '-', 'max_sample': '-', \
            'min': '-',  'min_sample': '-'}
        if opt['hmv'] != 'None':
            if opt['hidesat']:
                hmvdf = vio.copy()
            elif opt['hidevio']:
                hmvdf = sat.copy()
            else:
                hmvdf = pd.concat([sat, vio], axis = 0)
            hmvdf = hmvdf[opt['hmv']]
            norm = matplotlib.colors.Normalize(vmin = hmvdf.min(), vmax = hmvdf.max())
            if opt['mycmap'] != None:
                if type(opt['mycmap']) == str:
                    cmap = plt.cm.get_cmap(opt['mycmap'], 256)
                else:
                    cmap = opt['mycmap']
            else:
                cmap = self.cmap_opt[opt['cmap']]
            sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
            
            # ----- Calculating heat map variable values in design space ----- #
            hmv_R['mean']  = sat[opt['hmv']].mean()
            hmv_R['max']  = sat[opt['hmv']].max()
            hmv_R['max_sample'] = sat[sat[opt['hmv']] == hmv_R['max']]
            hmv_R['min']  = sat[opt['hmv']].min()
            hmv_R['min_sample'] = sat[sat[opt['hmv']] == hmv_R['min']]
        self.report.update({'hmv': hmv_R})
            
        # ------------------------------ Plotting ------------------------------ #
        # Check size of each dataframe
        nosat_flag = False
        novio_flag = False
        if sat.size == 0:
            nosat_flag = True
            print('No samples satisfied all constraints.')
        if vio.size == 0:
            novio_flag = True
            print('All samples satisfied all constraints.')
        
        # Check plot dimension
        dim = len(vnames)
        if dim == 2:   # 2D plot
            fig, ax = plt.subplots(figsize = opt['fs'])
        elif dim == 3: # 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d', computed_zorder = opt['czorder'])
        elif dim > 3:
            print('Dimension of vnames is larger than 3.')
        self.ax = ax
        self.fig = fig
        
        # Scatter plot        
        if opt['hidesat'] == False:
            if nosat_flag == False:
                if (opt['hmv'] == 'None') or (opt['hidehmv'] == True):
                    ax.scatter(*zip(*sat[vnames].to_numpy()), marker = opt['satmarker'],\
                        facecolors = opt['satfill'], label = opt['satlabel'],\
                        color = opt['satcolor'], alpha = opt['alpha'],\
                        zorder = opt['satzorder'])
                else:
                    fig.colorbar(sm, label = opt['hmvlabel'],\
                        location = opt['cbarloc'], orientation = opt['cbaror'], pad = opt['cbarpad'])
                    ax.scatter(*zip(*sat[vnames].to_numpy()), marker = opt['satmarker'],\
                        label = opt['satlabel'], color = cmap(norm(sat[opt['hmv']])),\
                        alpha = opt['alpha'], zorder = opt['satzorder'])                
        if opt['hidevio'] == False:
            if novio_flag == False:
                if (opt['hmv'] == 'None') or (opt['hidehmv'] == True):
                    ax.scatter(*zip(*vio[vnames].to_numpy()), marker = opt['viomarker'],\
                        facecolors = opt['viofill'], label = opt['violabel'],\
                        color = opt['viocolor'], alpha = opt['alpha'],\
                        zorder = opt['viozorder'])
                else:
                    ax.scatter(*zip(*vio[vnames].to_numpy()), marker = opt['viomarker'],\
                        label = opt['violabel'], color = cmap(norm(vio[opt['hmv']])),\
                        alpha = opt['alpha'], zorder = opt['viozorder'])
        
        # ----- Design space surface/boundary ----- #
        if opt['hidedsp'] == False:
            if self.space_size == None:
                self.shp = self.find_DSp(vnames, opt = opt)
            shp = self.shp
            if dim == 2:
                plt.plot(*zip(*shp['edges_val']),\
                    linewidth = opt['dspwidth'], linestyle = opt['dspstyle'],\
                    color = opt['dspcolor'], label = opt['dsplabel'],\
                    zorder = opt['dspzorder'])

            if dim == 3:
                surf = ax.plot_trisurf(*zip(*shp['P']), triangles = shp['tri'], \
                    color = opt['dspcolor'],\
                    alpha = opt['dspalpha'], label = opt['dsplabel'],\
                    zorder = opt['dspzorder'])
                surf._facecolors2d=surf._facecolor3d
                surf._edgecolors2d=surf._edgecolor3d
        
        # Limits for axes
        if opt['axeslimdf'] != 'best':
            if opt['axeslimdf'] == 'sat':
                axesdf = self.sat.copy()
            elif opt['axeslimdf'] == 'vio':
                axesdf = self.vio.copy()
            elif opt['axeslimdf'] == 'df':
                axesdf = self.df.copy()
            vmax = axesdf[vnames].max().to_numpy()
            vmin = axesdf[vnames].min().to_numpy()
            vrange = vmax - vmin
            limfactor = opt['limfactor']
            if dim == 2:
                plt.xlim([vmin[0] - limfactor*vrange[0], vmax[0] + limfactor*vrange[0]])
                plt.ylim([vmin[1] - limfactor*vrange[1], vmax[1] + limfactor*vrange[1]])
            if dim == 3:
                ax.set_xlim([vmin[0] - limfactor*vrange[0],\
                    vmax[0] + limfactor*vrange[0]])
                ax.set_ylim([vmin[1] - limfactor*vrange[1],\
                    vmax[1] + limfactor*vrange[1]])
                ax.set_zlim([vmin[2] - limfactor*vrange[2],\
                    vmax[2] + limfactor*vrange[2]])
        
        
        # Labels
        plt.xlabel(opt['xlabel'])
        plt.ylabel(opt['ylabel'])
        if dim == 3:
            ax.set_zlabel(opt['zlabel'])
            ax.view_init(elev = opt['elev'], azim = opt['azim'])
        if (opt['hmv'] == 'None') or (opt['hidehmv'] == True):
            plt.legend(loc = opt['legloc'],\
                framealpha = opt['framealpha']).set_zorder(opt['legendzorder'])
        
        # Saving
        plt.tight_layout()
        if opt['save_flag']:
            plt.savefig(opt['save_name'], dpi = opt['save_dpi'])
        return self.report
    
    def find_DSp(self, vnames = None, opt = {}):
        """
        Create a hull using alphashape
        Depending on alpha (opt['a']) value it can be either convex/concave
        # ----- Hull Parameters ----- #
        'a': None, # Alpha value -> at large alpha, hull becomes convex
        If None: use the product of axis bounds range
        'amul': 2, # Alpha multiplier value (wrt to product of mean axes)
        """
        import numpy as np
        import pandas as pd
        from time import time
        start_t = time()

        sat = self.sat
        if vnames == None:
            vnames = self.vnames
        else:
            self.vnames = vnames
        points = sat[vnames].to_numpy(dtype='float')
        vio = self.vio
        vpoints = vio[vnames].to_numpy(dtype='float')
        self.opt.update({'vnames': vnames})
        self.opt.update(opt) 
        print_flag = self.opt['printF']
        opt = self.opt
        extra_points = opt['extra_points']
        if len(extra_points) != 0:
            points = np.vstack([points, extra_points])
            self.extra_points = extra_points
        a = opt['a']
        if a == None:
            fdf = pd.concat([sat, vio], axis = 0)[vnames]
            a = np.product(fdf[vnames].max() - fdf[vnames].min())

        dim = len(vnames)
        if dim == 2:
            find_shp = self.alphashape_2D
            inside = self.inside2D
        elif dim == 3:
            find_shp = self.alphashape_3D
            inside = self.inside3D

        if opt['opt_amul'] == False: # Single run
            amul = opt['amul']
            alpha = a*amul
            self.alpha = alpha
            shp = find_shp(points, alpha)
            shp['amul'] = amul
            sol_flag = f'No optimisation performed'
            # opt_log = 'N/A'

        elif opt['opt_amul'] == True: # Use bisection
            # opt_log = []
            tol = opt['tol']
            lb = opt['lb']
            ub = opt['ub']
            maxiter = opt['maxiter']
            # Bisection algorithm
            for i in range(maxiter):
                mp = lb + (ub - lb)/2
                shp = find_shp(points, a*mp)
                shp['amul'] = mp
                shp['iter'] = i + 1
                # opt_log.append(shp)
                self.shp = shp
                r = inside(vpoints, shp)
                flag = True in r
                vnum = vio[r].shape[0]
                maxvnum = opt['maxvp']*(self.sat.shape[0] + vnum)
                if vnum <= maxvnum: # tolerance based
                    flag = False
                else:
                    flag = True
                if flag == True:
                    ub = mp
                elif flag == False:
                    lb = mp
                    if ub - lb <= tol:
                        if print_flag:
                            print(i + 1, mp, flag)
                        sol_flag = f'[{i + 1}] Optimal amul found: {mp:6f}   alpha: {a*mp:6f}   Tol: {tol:1.3e}   Gap: {ub - lb:1.3e}\nvnum: {vnum}   maxvnum: {maxvnum}'
                        break
                if print_flag:
                    print(i + 1, mp, flag)
            if (i + 1) == maxiter:
                sol_flag = f'Max iterations reached: {maxiter} iterations  amul: {mp:6f}'
        if print_flag:
            print(sol_flag)

        # Calculate volume
        if dim == 3:
            volume = 0
            for v in shp['P'][shp['tetras']]:
                volume += np.abs(np.dot(v[0] - v[3], np.cross(v[1] - v[3], v[2] - v[3])))/6
            shp['size'] = volume
        self.vindsp = vio[inside(vpoints, shp)]
        self.indsp = pd.concat([sat, self.vindsp]).reset_index()
        space_size = shp['size']
        self.shp = shp
        self.space_size = space_size
        self.bF = shp['tri']
        self.P = shp['P']
        self.report['space_size'] = space_size
        self.report['sol_flag'] = sol_flag
        # self.report['opt_log'] = opt_log
        
        end_t = time()
        comp_t = end_t - start_t
        self.report['time'] = comp_t
        return shp
    
    def alphashape_2D(self, P, alpha):
        """
        Calculate the alphashape boundary from a point cloud P (2D np array)
        Adapted from Kostas Markakis (https://stackoverflow.com/users/10105748/kostas-markakis)
        https://stackoverflow.com/a/62951837
        """
        from shapely.geometry import MultiLineString
        from shapely.ops import unary_union, polygonize
        from scipy.spatial import Delaunay
        from collections import Counter
        import numpy as np
        import itertools

        v = Delaunay(P).vertices
        # Calculate the sides of the triangles
        a = (P[v][:, 0][:, 0] - P[v][:, 1][:, 0])**2 + \
            (P[v][:, 0][:, 1] - P[v][:, 1][:, 1])**2
        b = (P[v][:, 1][:, 0] - P[v][:, 2][:, 0])**2 + \
            (P[v][:, 1][:, 1] - P[v][:, 2][:, 1])**2
        c = (P[v][:, 2][:, 0] - P[v][:, 0][:, 0])**2 + \
            (P[v][:, 2][:, 1] - P[v][:, 0][:, 1])**2
        a, b, c = np.sqrt([a, b, c])
        s = (a + b + c)*0.5
        area = np.sqrt(s*(s - a)*(s - b)*(s - c)) # area from Heron's formula
        alpha_filter = a*b*c/(4.0*area) < alpha

        # Filter vertices of alpha shape based on alpha radius
        tri = v[alpha_filter]
        # edges = np.vstack([edges[:, 0:2], edges[:, 1:3], np.delete(edges, 2, 1)])
        # edges = list(zip(edges[:, 0], edges[:, 1]))
        edges = [tuple(sorted(cb)) for e in tri for cb in itertools.combinations(e, 2)]

        count = Counter(edges)
        # Keep only edges that appear one time (concave hull edges)
        edges = [e for e, c in count.items() if c == 1]
        edges_val = [(P[e[0]], P[e[1]]) for e in edges]

        # Return points in order for plotting
        ml = MultiLineString(edges_val)
        poly = polygonize(ml)
        hull = unary_union(list(poly))
        hull_vertices = hull.exterior.coords.xy
        h = np.zeros((len(hull_vertices[0]), 2))
        h[:, 0] = hull_vertices[0]
        h[:, 1] = hull_vertices[1]

        shp = {}
        shp['P'] = P
        shp['verts'] = np.unique(edges)
        shp['tri'] = tri
        shp['tetras'] = None
        shp['edges'] = edges
        shp['edges_val'] = h
        shp['alpha'] = alpha

        # Area of alpha shape
        v = tri
        a = (P[v][:, 0][:, 0] - P[v][:, 1][:, 0])**2 + \
            (P[v][:, 0][:, 1] - P[v][:, 1][:, 1])**2
        b = (P[v][:, 1][:, 0] - P[v][:, 2][:, 0])**2 + \
            (P[v][:, 1][:, 1] - P[v][:, 2][:, 1])**2
        c = (P[v][:, 2][:, 0] - P[v][:, 0][:, 0])**2 + \
            (P[v][:, 2][:, 1] - P[v][:, 0][:, 1])**2
        a, b, c = np.sqrt([a, b, c])

        s = (a + b + c)*0.5
        area = np.sqrt(s*(s - a)*(s - b)*(s - c)) # area from Heron's formula
        shp['size'] = np.sum(area)
        return shp

    def alphashape_3D(self, P, alpha):
        """
        Calculate the alphashape boundary from a point cloud P (3D np array)
        Adapted from Geun (https://stackoverflow.com/users/9091202/geun)
        https://stackoverflow.com/a/58113037

        Compute the alpha shape (concave hull) of a set of 3D points.
        Parameters:
            pos - np.array of shape (n,3) points.
            alpha - alpha value.
        return
            outer surface vertex indices, edge indices, and triangle indices
        """
        from scipy.spatial import Delaunay
        import numpy as np
        import pandas as pd
        tetra = self.tetra
        r = self.r
        if tetra == None:
            tetra = Delaunay(P)
            # Find radius of the circumsphere.
            # By definition, radius of the sphere fitting inside the tetrahedral needs 
            # to be smaller than alpha value
            # http://mathworld.wolfram.com/Circumsphere.html

            tetrapos = np.take(P, tetra.vertices,axis=0)
            normsq = np.sum(tetrapos**2, axis=2)[:, :, None]
            ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))

            a  =  np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
            a[a == 0] = 1e-30
            Dx =  np.linalg.det(np.concatenate((normsq, tetrapos[:,:,[1,2]], ones), axis = 2))
            Dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:,:,[0,2]], ones), axis = 2))
            Dz =  np.linalg.det(np.concatenate((normsq, tetrapos[:,:,[0,1]], ones), axis = 2))
            c  =  np.linalg.det(np.concatenate((normsq, tetrapos), axis = 2))
            r  =  np.sqrt(Dx**2 + Dy**2 + Dz**2 - 4*a*c)/(2*np.abs(a))
            self.tetra = tetra
            self.r = r
        # Find tetrahedrals
        tetras = tetra.vertices[r<alpha, :]

        # triangles
        triComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
        triangles = tetras[:, triComb].reshape(-1, 3)
        triangles = np.sort(triangles, axis = 1)

        # Remove triangles that occurs twice, because they are within shapes
        triangles = pd.DataFrame(triangles).drop_duplicates(keep = False).to_numpy()

        #edges
        edgeComb = np.array([(0, 1), (0, 2), (1, 2)])
        edges = triangles[:, edgeComb].reshape(-1, 2)
        edges = np.sort(edges, axis = 1)
        edges = np.unique(edges, axis = 0)

        vertices = np.unique(edges)

        shp = {}
        shp['P'] = P
        shp['verts'] = vertices
        shp['tri'] = triangles
        shp['edges'] = edges
        shp['tetras'] = tetras
        shp['edges_val'] = None
        shp['alpha'] = alpha

        # volume = 0
        # for v in P[tetras]:
        #     volume += np.abs(np.dot(v[0] - v[3], np.cross(v[1] - v[3], v[2] - v[3])))/6
        shp['size'] = 420e-10
        return shp

    def collect_frames(self, anielev = 10, sfolder = 'animation', sname = 'plot',\
         leading_no = 0, sdpi = 100):
        """
        ONLY FOR 3D PLOTS
        Collect .png images for animation of plot by rotating the 3D plot
        720 images will be collected with 0.5 increments for a total of 1 whole rotation
        Args:
            ax ([type]): matplotlib axes
            anielev ([type]): elevation of the figure
            sfolder ([type]): string of the save folder
            sname ([type]): save name of the pictures generated, 
            will be followed by {i:04}.png
            sdpi (int, optional): [saved picture dpi]. Defaults to 100.
        """
        import matplotlib.pyplot as plt
        # Create save folder
        import os
        try:
            os.makedirs(sfolder)
        except FileExistsError:
            print('Folder already exist')
            
        ax = self.ax
        ax.view_init(elev = anielev, azim = 0)
        for ii in range(0, 2*360, 1):
                ax.azim += 0.5
                plt.savefig(f'{sfolder}{sname}_frame_{leading_no + ii:04}.png',\
                    dpi = sdpi)
    def inside2D(self, x, shp = None):
        """
        Returns False if x is not in self.shp, True otherwise (2D)
        x: list of [x, y], or [[x1, y1], [x2, y2], ...]
        """
        import numpy as np
        from shapely.geometry import Point, Polygon
        if shp == None:
            shp = self.shp
        poly = Polygon(shp['edges_val'])
        dim = len(np.array(x).shape)

        if dim == 1:
            x = [x]
        vP = [Point(i[0], i[1]) for i in x]
        out = [poly.contains(p) for p in vP]

        if dim == 1:
            out = out[0]
        return out
    
    def inside3D(self, x, shp = None):
        """
        Returns False if x is not in self.shp, True otherwise (3D)
        x: list of [x, y, z], or [[x1, y1, z1], [x2, y2, z2], ...]
        https://stackoverflow.com/a/57901916
        https://stackoverflow.com/a/41851137/12056867
        """
        import numpy as np
        if shp == None:
            shp = self.shp
        dim = len(np.array(x).shape)

        no_x = np.array(x).shape[0]
        if dim == 2:
            no_splits = self.opt['no_splits']
            if no_x < no_splits:
                no_splits = int(no_x/2)
            x_list = np.array_split(x, no_splits)
        else:
            x_list = [x]

        # Find which tetrahedron the point lies in
        node_coordinates = shp['P']
        node_ids = shp['tetras']

        res_list = []
        for x_s in x_list:
            p = np.array(x_s)

            ori = node_coordinates[node_ids[:, 0],:]
            v1 = node_coordinates[node_ids[:, 1],:] - ori
            v2 = node_coordinates[node_ids[:, 2],:] - ori
            v3 = node_coordinates[node_ids[:, 3],:] - ori
            n_tet = len(node_ids)
            v1r = v1.T.reshape((3, 1, n_tet))
            v2r = v2.T.reshape((3, 1, n_tet))
            v3r = v3.T.reshape((3, 1, n_tet))
            mat = np.concatenate((v1r, v2r, v3r), axis=1)
            inv_mat = np.linalg.inv(mat.T).T
            if p.size == 3:
                p = p.reshape((1,3))
            n_p = p.shape[0]
            orir = np.repeat(ori[:,:,np.newaxis], n_p, axis=2)
            newp = np.einsum('imk,kmj->kij',inv_mat,p.T-orir)
            val = np.all(newp>=0, axis=1) & np.all(newp <=1, axis=1) & (np.sum(newp, axis=1)<=1)
            id_tet, id_p = np.nonzero(val)
            res = -np.ones(n_p, dtype=id_tet.dtype) # Sentinel value
            res[id_p] = id_tet
            res_list.append(res)
        res = np.concatenate(res_list)

        if dim == 1:
            x = [x]
        # return res
        out = []
        for i, r in enumerate(res):
            V = shp['P'][shp['tetras'][r]]
            p = x[i]
            # Find the transform matrix from orthogonal to tetrahedron system
            v1 = V[1]-V[0] ; v2 = V[2]-V[0] ; v3 = V[3]-V[0]
            mat = np.array((v1,v2,v3)).T
            # mat is 3x3 here
            M1 = np.linalg.inv(mat)
            # apply the transform to P (v1 is the origin)
            newp = M1.dot(p - V[0])
            # perform test
            out.append(np.all(newp>=0) and np.all(newp <=1) and np.sum(newp)<=1)
        if dim == 1:
            out = out[0]
        return out

    def check_point(self, x):
        """
        Checks whether point x lies within the hull or not.
        """
        import numpy as np
        from time import time
        start_t = time()
        vn = self.vnames
        opt = self.opt
        sat = self.sat
        df = self.df

        # Bisection params
        pclb = opt['AORlb']
        pcub = opt['AORub']
        tol = opt['AORtol']
        maxiter = opt['AORmaxiter']
        print_flag = opt['AORprintF']
        AORopt_log = []

        dim = len(vn)
        if dim == 2:
            inside = self.inside2D
        elif dim == 3:
            inside = self.inside3D

        inputs_max = df[vn].max().to_numpy()
        inputs_min = df[vn].min().to_numpy()
        inputs_range = inputs_max - inputs_min


        not_in_region_flag = False
        if inside(x) == False:
            print('x is not inside DSp.')
            not_in_region_flag = True
            fs_R = {'x': x, 'FR': 'N/A', 'rmax': 'N/A', 'rmin': 'N/A',
            'space_size': 'N/A', 'plusmin': 'N/A', 'nosam': 'N/A', 
            'hmv': 'N/A', 'hmv_sam_flag': 'N/A', 'not_in_region_flag': not_in_region_flag}
        else: # Bisection
            for i in range(maxiter):
                pc = pclb + (pcub - pclb)/2
                gap = pcub - pclb
                if dim == 2:
                    verts = np.array(
                            [[x[0] - pc*inputs_range[0], x[1] - pc*inputs_range[1]],
                            [x[0] - pc*inputs_range[0], x[1] + pc*inputs_range[1]],
                            [x[0] + pc*inputs_range[0], x[1] + pc*inputs_range[1]],
                            [x[0] + pc*inputs_range[0], x[1] - pc*inputs_range[1]]]
                                )
                elif dim == 3:
                    verts = np.array(
                            [[x[0] - pc*inputs_range[0],\
                            x[1] - pc*inputs_range[1], x[2] - pc*inputs_range[2]],
                            [x[0] + pc*inputs_range[0],\
                            x[1] - pc*inputs_range[1], x[2] - pc*inputs_range[2]],
                            [x[0] - pc*inputs_range[0],\
                            x[1] + pc*inputs_range[1], x[2] - pc*inputs_range[2]],
                            [x[0] - pc*inputs_range[0],\
                            x[1] - pc*inputs_range[1], x[2] + pc*inputs_range[2]],
                            [x[0] + pc*inputs_range[0],\
                            x[1] + pc*inputs_range[1], x[2] - pc*inputs_range[2]],
                            [x[0] + pc*inputs_range[0],\
                            x[1] + pc*inputs_range[1], x[2] + pc*inputs_range[2]],
                            [x[0] + pc*inputs_range[0],\
                            x[1] - pc*inputs_range[1], x[2] + pc*inputs_range[2]],
                            [x[0] - pc*inputs_range[0],\
                            x[1] + pc*inputs_range[1], x[2] + pc*inputs_range[2]]]
                                )
                flag = False in [inside(verts[i]) for i in range(verts.shape[0])]
                AORopt_log.append({'iter': i+1, 'pc': pc, 'gap': gap, 'flag': flag, 'verts': verts})
                if flag == False:
                    pclb = pc
                    if gap <= tol:
                        if print_flag:
                            print(i+1, flag, gap)
                        break
                elif flag == True:
                    pcub = pc
                if print_flag:
                    print(i+1, flag, gap)

            if dim == 2:
                verts = verts.tolist()
                verts.append(verts[0])
                FR = np.array(verts)
            elif dim == 3:
                FR = [[verts[0], verts[1], verts[4], verts[2], verts[0]],
                    [verts[0], verts[3], verts[6], verts[1], verts[0]],
                    [verts[3], verts[7], verts[5], verts[6], verts[3]], 
                    [verts[6], verts[1], verts[4], verts[5], verts[6]],
                    [verts[7], verts[3], verts[0], verts[2], verts[7]],
                    [verts[2], verts[4], verts[5], verts[7], verts[2]]]

            # ----- KPIs ----- #
            rmax = np.array(verts).max(axis = 0)
            rmin = np.array(verts).min(axis = 0)
            fs_size = (rmax - rmin).prod()
            plusmin = (rmax - rmin)/2

            # ----- Heat map variable ----- #
            hmv_fs_R = {'name': opt['hmv'], 'mean': '-', 'max': '-', 'max_sample': '-', 
            'min': '-',  'min_sample': '-', 'fs_all_samples': '-'}
            no_samples_flag = False
            fs_df = df.copy()
            if opt['hmv'] != 'None':
                fs_df = sat.copy()
                for i in range(dim):
                    fs_df = fs_df[fs_df[vn[i]] <= rmax[i]]
                for i in range(dim):
                    fs_df = fs_df[fs_df[vn[i]] >= rmin[i]]
                if fs_df.shape[0] == 0:
                    print('No samples inside AOR available.')
                    no_samples_flag = True
                if no_samples_flag == False:
                    hmv_fs_R['mean']       = fs_df[opt['hmv']].mean()
                    hmv_fs_R['max']        = fs_df[opt['hmv']].max()
                    hmv_fs_R['max_sample'] = fs_df[fs_df[opt['hmv']] == hmv_fs_R['max']]
                    hmv_fs_R['min']        = fs_df[opt['hmv']].min()
                    hmv_fs_R['min_sample'] = fs_df[fs_df[opt['hmv']] == hmv_fs_R['min']]
                    hmv_fs_R['fs_all_samples'] = fs_df
            fs_R = {'x': x, 'FR': FR, 'rmax': rmax, 'rmin': rmin, 'space_size': fs_size, 
                'plusmin': plusmin, 'nosam': fs_df.shape[0], 'hmv': hmv_fs_R, 
                'hmv_sam_flag': no_samples_flag, 'not_in_region_flag': not_in_region_flag, 'AORopt_log': AORopt_log, 'fs_df': fs_df}
        
        end_t = time()
        comp_t = end_t - start_t
        fs_R.update({'time': comp_t})
        self.inAOR = fs_df
        self.all_x.update({str(x): fs_R})
        return fs_R
        
    def find_AOR(self, x):
        """
        Plot the AOR
        """
        import matplotlib.pyplot as plt
        
        ax = self.ax
        opt = self.opt
        vnames = self.vnames
        dim = len(vnames)
        
        # ----- Use self.check_point to calculate AOR ----- #
        if str(x) not in self.all_x.keys():
            fs_R = self.check_point(x)
        else:
            fs_R = self.all_x[str(x)]
            if fs_R['not_in_region_flag']:
                print('x is not inside DSp.')
        self.inAOR = fs_R['fs_df']
        FR = fs_R['FR']
        not_in_region_flag = fs_R['not_in_region_flag']
        
        # ----- Plotting ----- #
        ax.scatter(*zip(x), s = opt['nopsize'], marker = opt['nopmarker'],\
            color = opt['nopcolor'], label = opt['noplabel'],\
            linewidths = opt['nopwidth'], zorder = opt['nopzorder'])
        if not_in_region_flag:
            pass
        else:
            # ----- 2D space ----- #
            if dim == 2:
                plt.plot(*zip(*FR), linestyle = opt['aorstyle'], color = opt['aorcolor'],\
                    linewidth = opt['aorwidth'], label = opt['aorlabel'],\
                    zorder = opt['aorzorder'])
            # ----- 3D space ----- #
            if dim == 3: 
                for i in FR:
                    plt.plot(*zip(*i[:-1]), color = opt['aorcolor'],\
                        zorder = opt['aorzorder'])
                plt.plot(*zip(*FR[-1]), color = opt['aorcolor'], label = opt['aorlabel'],\
                    zorder = opt['aorzorder'])
        
        if (opt['hmv'] == 'None') or (opt['hidehmv'] == True):
            plt.legend(loc = opt['legloc'],\
                framealpha = opt['framealpha']).set_zorder(opt['legendzorder'])
        
        return fs_R
    
    def send_output(self, output_filename = 'DSI_output', appendix = False,\
            rp_pkl = False):
        """
        Send report of the DSI study as a txt file and .pkl file.
        """
        rp = self.report
        report_pkl = rp.copy()
        vnames = self.vnames
        f = open(f'{output_filename}.txt', 'w')
        # Headers
        f.write('Design Space Identification\n')
        f.write(f'Dataset name: {output_filename}\n\n')
        f.write(f'No of samples: {self.df.shape[0]}\n')

        f.write('Variables/parameters varied: \n')
        for i, name in enumerate(vnames):
            f.write(f'{name}\n')

        f.write('\nConstraints used: \n')
        for i in self.constraints:
            
            f.write(f'{i:10}   Lower bound: {self.constraints[i][0]:10}'+\
                f'   Upper bound: {self.constraints[i][1]:10}\n')


        # DSI results
        f.write(f'\n# -------------------------- RESULTS -------------------------- #\n')
        f.write(f'Design space size: {rp["space_size"]:10f}\n')
        f.write(f'Number of samples in DSp: {self.sat.shape[0]}\n\n')
        f.write(f'Average {rp["hmv"]["name"]}:    {rp["hmv"]["mean"]:10f}\n')
        f.write(f'DS Maximum {rp["hmv"]["name"]}: {rp["hmv"]["max"]:10f}\n')
        f.write(f'DS Minimum {rp["hmv"]["name"]}: {rp["hmv"]["min"]:10f}\n')
        f.write(f'\n-----------------------------------------------------------------\n')
        f.write(f'Detailed maximum point: \n')
        if type(rp["hmv"]["max_sample"]) != str:
            f.write(f'{rp["hmv"]["max_sample"].to_string()}\n\n')
        f.write(f'Detailed minimum point: \n')
        if type(rp["hmv"]["min_sample"]) != str:
            f.write(f'{rp["hmv"]["min_sample"].to_string()}')
        f.write(f'\n-----------------------------------------------------------------\n')

        if len(self.all_x) != 0:
            for n, x_i in enumerate(list(self.all_x.keys())):
                rep = self.all_x[x_i]
                report_pkl['x'] = rep
                x = rep['x']
                rmax = rep['rmax']
                rmin = rep['rmin']
                fs_size = rep['space_size']
                plusmin = rep['plusmin']

                if x != None:
                    f.write(f'\n\n\n# ------------------------------ Acceptable ' + \
                    f'Operating Region {n+1:03} ------------------------------ #\n')
                    f.write(f'AOR point: \n')
                    if rep['not_in_region_flag']:
                        for i, vn in enumerate(vnames):
                            f.write(f'{vn}: {x[i]:10}\n')
                        f.write(f'-------------------------------------------------\n')
                        f.write(f'------------ POINT IS NOT IN THE DSp ------------\n')
                        f.write(f'-------------------------------------------------\n\n')
                    else:
                        for i, vn in enumerate(vnames):
                            f.write(f'{vn}: {x[i]:10f}' + ' +- ' +\
                                f'{plusmin[i]:10f} Range: {rmin[i]:5f} - {rmax[i]:5f}' +\
                                        f' ({rmax[i] - rmin[i]:10})\n')
                        f.write(f'AOR size: {fs_size:10f}\n')

                        if rep['hmv_sam_flag']:
                            f.write('No samples inside AOR available.')
                        else:
                            f.write(f'\nNumber of samples inside AOR:' +\
                                f'{rep["nosam"]}\n')
                            f.write(f'Average {rep["hmv"]["name"]}:' +\
                                f'{rep["hmv"]["mean"]:10f}\n')
                            f.write(f'DS Maximum {rep["hmv"]["name"]}:' +\
                                f'{rep["hmv"]["max"]:10f}\n')
                            f.write(f'DS Minimum {rep["hmv"]["name"]}:' +\
                                f'{rep["hmv"]["min"]:10f}\n')
                            f.write(f'\n-----------------------------------------------'+\
                                '--------------------------------------------------\n')
                            f.write(f'Detailed maximum point: \n')
                            if type(rep["hmv"]["max_sample"]) != str:
                                f.write(f'{rep["hmv"]["max_sample"].to_string()}\n\n')
                            f.write(f'Detailed minimum point: \n')
                            if type(rep["hmv"]["min_sample"]) != str:
                                f.write(f'{rep["hmv"]["min_sample"].to_string()}')
                            f.write(f'\nAll samples inside AOR: \n')
                            if type(rep["hmv"]["fs_all_samples"]) != str:
                                f.write(f'{rep["hmv"]["fs_all_samples"].to_string()}')
                            f.write(f'\n-----------------------------------------------'+\
                                '--------------------------------------------------\n')

        if appendix:
            f.write('\n\n\n\n# ------------------------------ APPENDIX ----------------'+\
                '-------------- #\n')
            f.write(f'ALL SATISFIED SAMPLES: \n')
            f.write(f'\n{self.sat.to_string()}\n\n')
            f.write(f'\n---------------------------------------------------------------'+\
                '----------\n')
            f.write(f'ALL VIOLATED SAMPLES: \n')
            f.write(f'\n{self.vio.to_string()}\n\n')
            f.write(f'# ----- Envelope (hull) ----- #\n')
            f.write(f'Boundary Facets: \n')
            for i in self.bF:
                f.write(f'{str(i)}\n')
            f.write(f'\nPoints: \n')
            for i in self.P:
                f.write(f'{str(i)}\n')

        f.close()
        if rp_pkl:
            import pickle
            with open(output_filename + '.pkl', 'wb') as handle:
                pickle.dump(report_pkl, handle, protocol = pickle.HIGHEST_PROTOCOL)
            print(f'Pickle file saved at: {output_filename}.pkl')
        return report_pkl