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
    
def plot_DSp(shp, opt = {}, ax = None):
    """
    Takes in shp and plots the design space just like in normal DSI entity
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    dim = shp['P'].shape[1]

    # ---------------------------- Default options ---------------------------- #
        # Default options
    default_opt = {
        # ----- Plot Format ----- #
        'czorder': False,    # computed_zorder for 3D plots settings
        # ----- Design Space Parameters ----- #
        'dsplabel': 'DSp',   # Label of surface/boundary
        'dspcolor': 'black', # Color of the surface/boundary (both)
        'dspwidth': 4,       # Thickness of the boundary (2D)
        'dspstyle': '-',     # Line style of the boundary (2D)
        'dspalpha': 0.2,     # Transparency of surface/boundary (3D)
        'dspzorder': 20,     # To make sure it is plotted ontop of the samples
    }
    default_opt.update(opt)
    opt = default_opt

    if ax == None:
        if dim == 2:   # 2D plot
            fig, ax = plt.subplots()
        elif dim == 3: # 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d', computed_zorder = opt['czorder'])
        elif dim > 3:
            print('Dimension of vnames is larger than 3.')
    if dim == 2:
        for i in range(len(shp['reg_bounds_val'])):
            if i == 0:
                ax.plot(*zip(*shp['reg_bounds_val'][i]),\
                    linewidth = opt['dspwidth'], linestyle = opt['dspstyle'],\
                    color = opt['dspcolor'], label = opt['dsplabel'],\
                    zorder = opt['dspzorder'], solid_capstyle = 'round')
            else:
                ax.plot(*zip(*shp['reg_bounds_val'][i]),\
                    linewidth = opt['dspwidth'], linestyle = opt['dspstyle'],\
                    color = opt['dspcolor'],\
                    zorder = opt['dspzorder'], solid_capstyle = 'round')

    if dim == 3:
        for i in range(len(shp['reg_bounds'])):
            if i == 0:
                surf = ax.plot_trisurf(*zip(*shp['P']), triangles = shp['reg_bounds'][i], \
                    color = opt['dspcolor'],\
                    alpha = opt['dspalpha'], label = opt['dsplabel'],\
                    zorder = opt['dspzorder'])
                surf._facecolors2d=surf._facecolor3d
                surf._edgecolors2d=surf._edgecolor3d
            else:
                surf = ax.plot_trisurf(*zip(*shp['P']), triangles = shp['reg_bounds'][i], \
                    color = opt['dspcolor'],\
                    alpha = opt['dspalpha'],\
                    zorder = opt['dspzorder'])
                surf._facecolors2d=surf._facecolor3d
                surf._edgecolors2d=surf._edgecolor3d
    return ax

def Sobol_sequence(lbd, ubd, power_no):
    """
    Create 2^power_no of inputs for sampling based on the lists of lbd (lower
    bound) and ubd (upper bound).
    """
    from scipy.stats import qmc
    sampler = qmc.Sobol(d = len(lbd), scramble = False)
    inputs = sampler.random_base2(m = power_no)
    inputs = qmc.scale(inputs, lbd, ubd)
    return inputs

def update_ds(ds):
    """
    Updates old ds options with the new ones
    """
    ds1 = DSI(ds.df)
    l1 = list(ds.default_opt.keys())
    l2 = list(ds1.default_opt.keys())
    l3 = l1 + l2
    up = []
    for i in l2:
        if l3.count(i) == 1:
            up.append(i)
    ds.opt.update()
    for u in up:
        ds.opt.update({u: ds1.default_opt[u]})
    return ds

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
    ds.screen(constraints)
    ds.plot(vnames, opt)
    if x != None:
        ds.find_AOR(x)
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
        df = df.copy()
        self.df = df
        self.report = {
            'nosam': df.shape[0], 'space_size': '-',
            'hmv': {'name': 'None', 'mean': '-', 'max': '-', 'max_sample': '-',\
                 'min': '-',  'min_sample': '-'},
        }
        self.all_x = {}
        self.space_size = None
        self.DTspreadsheet = None
        self.r = None
        
        # Set default color map for heat map plot
        inferno_modified = plt.get_cmap('inferno', 256)
        gray_modified    = plt.get_cmap('gray', 256)
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
            'xlabel': 'x',          # x axis label
            'ylabel': 'y',          # y axis label
            'zlabel': 'z',          # z axis label
            'labelpad': [8, 8, 8],  # Label padding
            'cmap': 'inferno80',    # Matplotlib colour map
            'hmv': 'None',          # heat map variable name
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
            'cbarshrink': 1.0,    # size of the colorbar
            'cbarfraction': 0.15, # fraction of original axes to use for colorbar
            'mycmap': 'viridis',  # Use your own cmap (input name of cmap as str)
            'cmapmax': None,       # max colorbar scale
            'cmapmin': None,       # min colorbar scale
            'cmapext': 'neither',  # if extended cbar is desired
            'cmapextmax': 'red',   # color of the "over max values"
            'cmapextmin': 'green', # color of the "under min values"
            # Satisfied samples
            'satlabel': 'Sat',     # Satisfied samples label
            'satcolor': '#FF9000', # Satisfied samples color
            'satmarker': '.',      # Marker of satisfied points
            'satmarkersize': 20,   # Marker size of violated points
            'satfill': '#FF9000',  # Marker fill color of satisfied points
            'satzorder': 5,        # Decides which level to be plotted on
            # Violated samples
            'violabel': 'Vio',     # Violated samples label
            'viocolor': '#005DC1', # Violated samples color
            'viomarker': '.',      # Marker of violated points
            'viomarkersize': 20,   # Marker size of violated points
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
            'lb': 1e-30,      # Lower bound of initial bisection run
            'ub': 5,          # Upper bound of initial bisection run
            'printF': True,   # If true, print iter details
            'maxvp': 0,       # Maximum allowed percentage of vio in DSp
        }
        self.opt = self.default_opt.copy()
        self.bw_template = {'alpha': 1, 'satcolor': 'gray', 'viocolor': 'black', \
            'satfill': 'gray', 'viofill': 'black', 'cmap': 'gray80',
                           'satmarker': 'o', 'viomarker': 'o'}
        self.normP = None
        return None
    
    def reset(self):
        """
        Reset options to default.
        """
        self.opt = self.default_opt.copy()
        return None
        
    def mm_norm(self, arr, normP):
        """
        Min max normalisation
        """ 
        arrmin = normP[0]
        arrmax = normP[1]
        return (arr - arrmin)/(arrmax - arrmin)

    def mm_rev(self, norm, normP):
        """
        Reverse min max normalisation
        """
        arrmin = normP[0]
        arrmax = normP[1]
        return norm*(arrmax - arrmin) + arrmin

    def mm_size_norm(self, size, normP):
        """
        Min max normalisation of space size
        """
        import numpy as np
        ss_norm = size/np.prod(normP[1] - normP[0])
        return ss_norm

    def mm_size_rev(self, ss_norm, normP):
        """
        Reverse min max normalisation of space size
        """
        import numpy as np
        size = ss_norm*np.prod(normP[1] - normP[0])
        return size
        
    def screen(self, constraints):
        """
        Takes in the DataFrame, data, and dictionary, constraints, 
        giving out the satisfied and violated DataFrame of samples
        constraints: {'output_name1': [lbd1, ubd1], 'output_name2': [lbd1, ubd2], ...}
        """
        import pandas as pd
        if 'SatFlag' in list(self.df.columns):
            self.df.pop('SatFlag')
        data = self.df.copy()
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
        
        # Update saved
        self.df['SatFlag'] = 'N/A'
        self.df.loc[sat.index, 'SatFlag'] = True
        self.df.loc[vio.index, 'SatFlag'] = False
        self.df['SatFlag'].astype('bool')
        return None
    
    def help(self, print_opt = False):
        """
        Prints usage instructions and ALL of the current options and return the 
        opt dictionary.
        """
        print('# ----- Usage Instructions ----- #')
        print('# 1. Create instance of ds with data from DataFrame df')
        print('ds = dside.DSI(df)')
        print('# 2. Screen the points using the constraints (dictionary)')
        print('ds.screen(constraints)')
        print('# 3. Find DSp boundaries based on vnames (list of variable names for the axes)')
        print('ds.find_DSp(vnames)')
        print('# 4. Plot the design space and the samples')
        print('ds.plot(vnames)')
        print('# 5. Plot the nominal point and AOR based on point x (list/numpy array)')
        print('ds.find_AOR(x)')
        print('# 6. Save the results in detailed output.txt file and output.pkl file')
        print("ds.send_output('output')")
        if print_opt:
            print('\n# ----- Options ----- #')
            for i in list(self.opt.keys()):
                print(f'{i:10}: {self.opt[i]}')
        return None
    
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
        opt = self.opt
        try:
            labelpad = opt['labelpad']
        except KeyError:
            labelpad = [8]*3
        if type(labelpad) == int:
            labelpad = [labelpad]
        if len(labelpad) == 1:
            labelpad *= 3
        plt.rcParams.update({'font.size': opt['font_size']})
            
        # ------------------------------ Plotting ------------------------------ #        
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
        ax, fig = self.scatter(vnames = None, opt = opt, ax = ax, fig = fig)
        
        # ----- Design space surface/boundary ----- #
        if opt['hidedsp'] == False:
            if self.space_size == None:
                self.find_DSp(vnames, opt = opt)
            shp = self.shp
            ax = self.plot_DSp(shp, opt = opt, ax = ax)
        
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
                ax.set_box_aspect(None, zoom = 0.85)
        
        
        # Labels
        plt.xlabel(opt['xlabel'], labelpad = labelpad[0])
        plt.ylabel(opt['ylabel'], labelpad = labelpad[1])
        if dim == 3:
            ax.set_zlabel(opt['zlabel'], labelpad = labelpad[2])
            ax.view_init(elev = opt['elev'], azim = opt['azim'])
        if (opt['hmv'] == 'None') or (opt['hidehmv'] == True):
            plt.legend(loc = opt['legloc'],\
                framealpha = opt['framealpha']).set_zorder(opt['legendzorder'])
        
        # Saving
        plt.tight_layout()
        self.ax = ax
        self.fig = fig
        if opt['save_flag']:
            plt.savefig(opt['save_name'], dpi = opt['save_dpi'])
        return None
    
    def plot_DSp(self, shp = 'default', opt = {}, ax = None):
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        if shp == 'default':
            shp = self.shp
        dim = shp['P'].shape[1]

        # ---------------------------- Internal definitions ---------------------------- #
        self.opt.update(opt)
        if self.opt['bw']:
            self.opt.update(self.bw_template)
        self.opt.update(opt)

        # ---------------------------- External definitions ---------------------------- #
        opt = self.opt

        if ax == None:
            if dim == 2:   # 2D plot
                fig, ax = plt.subplots(figsize = opt['fs'])
            elif dim == 3: # 3D plot
                fig = plt.figure()
                ax = fig.add_subplot(projection = '3d', computed_zorder = opt['czorder'])
            elif dim > 3:
                print('Dimension of vnames is larger than 3.')
            self.ax = ax
            self.fig = fig
        if dim == 2:
            for i in range(len(shp['reg_bounds_val'])):
                if i == 0:
                    plt.plot(*zip(*shp['reg_bounds_val'][i]),\
                        linewidth = opt['dspwidth'], linestyle = opt['dspstyle'],\
                        color = opt['dspcolor'], label = opt['dsplabel'],\
                        zorder = opt['dspzorder'], solid_capstyle = 'round')
                else:
                    plt.plot(*zip(*shp['reg_bounds_val'][i]),\
                        linewidth = opt['dspwidth'], linestyle = opt['dspstyle'],\
                        color = opt['dspcolor'],\
                        zorder = opt['dspzorder'], solid_capstyle = 'round')

        if dim == 3:
            for i in range(len(shp['reg_bounds'])):
                if i == 0:
                    surf = ax.plot_trisurf(*zip(*shp['P']), triangles = shp['reg_bounds'][i], \
                        color = opt['dspcolor'],\
                        alpha = opt['dspalpha'], label = opt['dsplabel'],\
                        zorder = opt['dspzorder'])
                    surf._facecolors2d=surf._facecolor3d
                    surf._edgecolors2d=surf._edgecolor3d
                else:
                    surf = ax.plot_trisurf(*zip(*shp['P']), triangles = shp['reg_bounds'][i], \
                        color = opt['dspcolor'],\
                        alpha = opt['dspalpha'],\
                        zorder = opt['dspzorder'])
                    surf._facecolors2d=surf._facecolor3d
                    surf._edgecolors2d=surf._edgecolor3d
        return ax
        
    def scatter(self, vnames = None, opt = {}, ax = None, fig = None):
        """
        Scatter plot of either 2D or 3D based on satisfied and violated points.
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
        self.opt.update(opt)
        if self.opt['bw']:
            self.opt.update(self.bw_template)
        self.opt.update(opt)
        
        # ---------------------------- External definitions ---------------------------- #
        sat = self.sat
        vio = self.vio
        opt = self.opt
        try:
            labelpad = opt['labelpad']
        except KeyError:
            labelpad = [8]*3
        if type(labelpad) == int:
            labelpad = [labelpad]
        if len(labelpad) == 1:
            labelpad *= 3
        plt.rcParams.update({'font.size': opt['font_size']})

        # ------------------------- Initialize figure and axes ------------------------- #
        # Check size of each dataframe
        nosat_flag = False
        novio_flag = False
        if sat.size == 0:
            nosat_flag = True
            print('No samples satisfied all constraints.')
        if vio.size == 0:
            novio_flag = True
            print('All samples satisfied all constraints.')
        if ax == None:
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
            cmapvmin = hmvdf.min()
            cmapvmax = hmvdf.max()
            if opt['cmapmax'] != None:
                cmapvmax = opt['cmapmax']
            if opt['cmapmin'] != None:
                cmapvmin = opt['cmapmin']
            norm = matplotlib.colors.Normalize(vmin = cmapvmin, vmax = cmapvmax)
            if opt['mycmap'] != None:
                if type(opt['mycmap']) == str:
                    cmap = plt.get_cmap(opt['mycmap'], 256)
                else:
                    cmap = opt['mycmap']
            else:
                cmap = self.cmap_opt[opt['cmap']]
            sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
            if opt['cmapext'] == 'both':
                sm.cmap.set_over(opt['cmapextmax'])
                sm.cmap.set_under(opt['cmapextmin'])
            elif opt['cmapext'] == 'max':
                sm.cmap.set_over(opt['cmapextmax'])
            elif opt['cmapext'] == 'min':
                sm.cmap.set_under(opt['cmapextmin'])
            
            # ----- Calculating heat map variable values in design space ----- #
            hmv_R['mean']  = sat[opt['hmv']].mean()
            hmv_R['max']  = sat[opt['hmv']].max()
            hmv_R['max_sample'] = sat[sat[opt['hmv']] == hmv_R['max']]
            hmv_R['min']  = sat[opt['hmv']].min()
            hmv_R['min_sample'] = sat[sat[opt['hmv']] == hmv_R['min']]
        self.report.update({'hmv': hmv_R})

        if opt['hidesat'] == False:
            if nosat_flag == False:
                if (opt['hmv'] == 'None') or (opt['hidehmv'] == True):
                    ax.scatter(*zip(*sat[vnames].to_numpy()), s = opt['satmarkersize'], marker = opt['satmarker'],\
                        facecolors = opt['satfill'], label = opt['satlabel'], color = opt['satcolor'], alpha = opt['alpha'], 
                        zorder = opt['satzorder'])
                else:
                    self.cbar = fig.colorbar(sm, ax = ax, label = opt['hmvlabel'], location = opt['cbarloc'], orientation = opt['cbaror'], pad = opt['cbarpad'], extend = opt['cmapext'], shrink = opt['cbarshrink'], fraction = opt['cbarfraction'])
                    if opt['cmapext'] == 'both':
                        self.cbar.cmap.set_over(opt['cmapextmax'])
                        self.cbar.cmap.set_under(opt['cmapextmin'])
                    elif opt['cmapext'] == 'max':
                        self.cbar.cmap.set_over(opt['cmapextmax'])
                    elif opt['cmapext'] == 'min':
                        self.cbar.cmap.set_under(opt['cmapextmin'])
                    ax.scatter(*zip(*sat[vnames].to_numpy()), s = opt['satmarkersize'], marker = opt['satmarker'], label = opt['satlabel'], color = cmap(norm(sat[opt['hmv']])), alpha = opt['alpha'], zorder = opt['satzorder'])                
        if opt['hidevio'] == False:
            if novio_flag == False:
                if (opt['hmv'] == 'None') or (opt['hidehmv'] == True):
                    ax.scatter(*zip(*vio[vnames].to_numpy()), s = opt['viomarkersize'], marker = opt['viomarker'],\
                        facecolors = opt['viofill'], label = opt['violabel'],\
                        color = opt['viocolor'], alpha = opt['alpha'],\
                        zorder = opt['viozorder'])
                else:
                    ax.scatter(*zip(*vio[vnames].to_numpy()), s = opt['viomarkersize'], marker = opt['viomarker'],\
                        label = opt['violabel'], color = cmap(norm(vio[opt['hmv']])),\
                        alpha = opt['alpha'], zorder = opt['viozorder'])
        return ax, fig

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
        
        if vnames == None:
            vnames = self.vnames
        else:
            self.vnames = vnames
            
        # Get normalization parameters (only used in calculations)
        normP = self.normP
        if normP == None:
            df = self.df
            normP = [df[vnames].min().to_numpy(), df[vnames].max().to_numpy()]
            self.normP = normP
        else:
            pass

        sat = self.sat
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
            a = np.prod(self.df[vnames].max())

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
            maxvp = opt['maxvp']
            if maxvp == 1:
                maxvnum = 1e30
            else:
                maxvnum = maxvp*self.sat.shape[0]/(1 - maxvp)
            if print_flag:
                print('Bisection search for alpha multiplier (radius)')
                print(f'    tol: {tol:3.2e}  maxiter: {maxiter}')
                print(f'    lb:  {lb:3.2e}  ub:      {ub:3.2e}')
                print(f'    maxvp: {maxvp:3.2e}  maxvnum: {maxvnum:.2f}')
                print(f'{92*"_"}')
                print(f'{"No iter":^10}|{"alpha multiplier":^20}|{"Violation Flag":^20}|{"Number vio inside":^20}|{"Bisection Gap":^20}')
                print(f'{92*"_"}')
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
                            print(f'{i + 1:^10}|{mp:^20.3e}|{str(flag):^20}|{vnum:^20}|{ub - lb:^20.3e}')
                        sol_flag = f'[{i + 1}] Optimal amul: {mp:1.4e}   alpha: {a*mp:1.3e}\nTol: {tol:1.3e}   Bisection Gap: {ub - lb:1.3e}\nvnum: {vnum}   maxvnum: {maxvnum:.2f}'
                        break
                if print_flag:
                    print(f'{i + 1:^10}|{mp:^20.3e}|{str(flag):^20}|{vnum:^20}|{ub - lb:^20.3e}')
            if (i + 1) == maxiter:
                sol_flag = f'Max iterations reached: {maxiter} iterations  amul: {mp:6f}'
        if print_flag:
            print(f'{92*"_"}')
            print(sol_flag)

        # Calculate design space size and classify regions
        if dim == 3:
            shp = self.classify_regions3D(shp)
            volume = 0
            for v in shp['P'][shp['simplices']]:
                volume += np.abs(np.dot(v[0] - v[3], np.cross(v[1] - v[3], v[2] - v[3])))/6
            shp['size'] = self.mm_size_norm(volume, self.normP)
        else:
            shp = self.classify_regions2D(shp)
            P = shp['P']
            simps = shp['simplices']
            v = simps
            a = (P[v][:, 0][:, 0] - P[v][:, 1][:, 0])**2 + \
                (P[v][:, 0][:, 1] - P[v][:, 1][:, 1])**2
            b = (P[v][:, 1][:, 0] - P[v][:, 2][:, 0])**2 + \
                (P[v][:, 1][:, 1] - P[v][:, 2][:, 1])**2
            c = (P[v][:, 2][:, 0] - P[v][:, 0][:, 0])**2 + \
                (P[v][:, 2][:, 1] - P[v][:, 0][:, 1])**2
            a, b, c = np.sqrt([a, b, c])
            s = (a + b + c)*0.5
            area = np.sqrt(s*(s - a)*(s - b)*(s - c)) # area from Heron's formula
            shp['size'] = self.mm_size_norm(np.sum(area), self.normP)
    
        if vpoints.shape[0] == 0:
            self.vindsp = None
            self.report['no_vindsp'] = 0
        else:
            self.vindsp = vio[inside(vpoints, shp)]
            self.report['no_vindsp'] = self.vindsp.shape[0]
        self.indsp = pd.concat([sat, self.vindsp]).reset_index()
        space_size = shp['size']
        self.shp = shp
        self.space_size = space_size
        self.bF = shp['simplices']
        self.P = shp['P']
        self.report['no_sat'] = sat.shape[0]
        self.report['no_vio'] = vio.shape[0]
        self.report['space_size'] = space_size
        self.report['no_reg'] = shp['no_reg']
        self.report['no_simps'] = shp['simplices'].shape[0]
        self.report['sol_flag'] = sol_flag
        # self.report['opt_log'] = opt_log
        if print_flag:
            print(f'{42*"="} Results {41*"="}')
            print(f'{"No. satisfied points":<32}: {sat.shape[0]}\n{"No. violated points":<32}: {vio.shape[0]}\n{"No. violated points inside DSp":<32}: {self.report["no_vindsp"]}')
            print(f'{"No. of regions":<32}: {shp["no_reg"]}\n{"No. of simplices":<32}: {shp["simplices"].shape[0]}')
            print(f'{"Size of design space":<32}: {space_size:1.3e}')
        
        end_t = time()
        comp_t = end_t - start_t
        self.report['time'] = comp_t
        return None
    
    def alphashape_2D(self, P, alpha):
        """
        Calculate the alphashape boundary from a point cloud P (2D np array)
        """
        from scipy.spatial import Delaunay
        import numpy as np
        import pandas as pd

        # Normalize points P
        normP = self.normP
        P = self.mm_norm(P, normP)

        spreadsheet = self.DTspreadsheet
        if type(spreadsheet) == type(None):
            # Get Delaunay triangulation of the points
            tri = Delaunay(P)
            simps = tri.simplices
            triP = P[simps]

            # --- Calculating circumcircle radius and center (vectorised) --- #
            PA = triP[:, 0, :]; Ax = PA[:, 0]; Ay = PA[:, 1]
            PB = triP[:, 1, :]; Bx = PB[:, 0]; By = PB[:, 1]
            PC = triP[:, 2, :]; Cx = PC[:, 0]; Cy = PC[:, 1]

            # translation of vertex A to the origin
            Bxo = Bx - Ax; Cxo = Cx - Ax # Axo = Ax - Ax; 
            Byo = By - Ay; Cyo = Cy - Ay # Ayo = Ay - Ay; 

            Do = (2*(Bxo*Cyo - Byo*Cxo)).clip(min = 1e-10) # denominator
            Uxo = (1/Do)*(Cyo*(Bxo**2 + Byo**2) - Byo*(Cxo**2 + Cyo**2))
            Uyo = (1/Do)*(Bxo*(Cxo**2 + Cyo**2) - Cxo*(Bxo**2 + Byo**2))
            R = np.sqrt(Uxo**2 + Uyo**2)          # radius of circumcircle
            U = np.array([Uxo + Ax, Uyo + Ay]).T  # center of circumcircle

            spreadsheet = pd.DataFrame(simps, columns = ['p1', 'p2', 'p3'])
            spreadsheet[['p1x', 'p1y']] = PA
            spreadsheet[['p2x', 'p2y']] = PB
            spreadsheet[['p3x', 'p3y']] = PC
            spreadsheet[['Ux', 'Uy']] = U
            spreadsheet['r'] = R

            self.DTspreadsheet = spreadsheet

        alpha_spreadsheet = spreadsheet[spreadsheet['r'] <= alpha].copy()

        # ----- Get edge lines only ----- #
        simps = alpha_spreadsheet[['p1', 'p2', 'p3']].to_numpy()
        edgeComb = np.array([(0, 1), (0, 2), (1, 2)]) # comb to separate tri to lines
        edges = simps[:, edgeComb].reshape(-1, 2) # separate tri to lines
        edges.sort(axis = 1) # sort vertex indices
        edges = pd.DataFrame(edges).drop_duplicates(keep = False).to_numpy() # get unique entries only

        shp = {
            'P': self.mm_rev(P, normP), 
            'simplices': simps, 
            'edges': edges, 
            'alpha': alpha,
            'alpha_spreadsheet': alpha_spreadsheet,
            'normalized_P': P,
            }
        return shp


    def classify_regions2D(self, shp = None):
        """
        Classify number of regions and their outer lines
        """
        import numpy as np
        import pandas as pd

        # Unpack shp
        if shp is None:
            shp = self.shp
        edges = shp['edges']
        P = shp['P']

        # ----- Identification of Regions and Ordering Boundary ----- #
        # Implemented 2D breadth-first search
        # Create worksheet
        ws = pd.DataFrame(edges[edges[:, 0].argsort()])
        ws['visit'] = False
        ws['region'] = np.nan

        change_init_flag = False
        reg = 0
        reg_bounds = [] # record ordered boundary vertices
        # Loop over every line to categorise them into regions
        while ws[ws['visit'] == False].shape[0] > 0:
            reg += 1

            # create sorted bounds
            bounds = []
            # find a point which have not been visited yet
            cur_idx = ws[ws['visit'] == False].index[0]
            ws.loc[cur_idx, 'visit'] = True  # set it to be visited
            ws.loc[cur_idx, 'region'] = reg  # categorise to region reg

            # taking both directions of the line to search neighbouring vertices
            current_vert1 = int(ws.loc[cur_idx][0])
            current_vert2 = int(ws.loc[cur_idx][1])

            # record boundary
            bounds.append(current_vert1)
            bounds.append(current_vert2)

            # setting flags for exploration of neighbouring vertices
            empty_v1 = False
            empty_v2 = False
            change_init_flag = False
            while not change_init_flag:  # while v1 and v2 are not empty
                # make working_sheet for easier manipulation
                w_s = ws[ws['visit'] == False].copy()

                # get v1 and v2 (contains current_vert1 and 2)
                v1 = pd.concat(
                    [w_s[w_s[0] == current_vert1], 
                     w_s[w_s[1] == current_vert1]]
                    )[[0, 1]]
                v2 = pd.concat(
                    [w_s[w_s[0] == current_vert2], 
                     w_s[w_s[1] == current_vert2]]
                    )[[0, 1]]

                # check if v1 is empty
                if v1.shape[0] != 0:
                    ws.loc[v1.index[0], 'visit'] = True  # set it to be visited
                    ws.loc[v1.index[0], 'region'] = reg  # categorise to region reg
                    next_vert1 = v1.T[v1.T != current_vert1].dropna()
                    if not next_vert1.empty:
                        current_vert1 = int(next_vert1.iloc[0, 0])
                        bounds.insert(0, current_vert1)  # record boundary
                    else:
                        empty_v1 = True
                else:
                    empty_v1 = True

                # check if v2 is empty
                if v2.shape[0] != 0:
                    ws.loc[v2.index[0], 'visit'] = True  # set it to be visited
                    ws.loc[v2.index[0], 'region'] = reg  # categorise to region reg
                    next_vert2 = v2.T[v2.T != current_vert2].dropna()
                    if not next_vert2.empty:
                        current_vert2 = int(next_vert2.iloc[0, 0])
                        bounds.append(current_vert2)  # record boundary
                    else:
                        empty_v2 = True
                else:
                    empty_v2 = True

                # if both v1 and v2 are empty, set change_init_flag to true
                if empty_v1 and empty_v2:
                    change_init_flag = True

            # Check if the last entry in bounds is the same as first
            if bounds[-1] != bounds[0]:
                if bounds[-2] == bounds[0]:  # check if second last is same as first
                    bounds = bounds[:-1]  # if yes, slice boundary
                else:  # else, there may be something wrong, print warning
                    print('WARNING: BOUNDARY FORMED MAY NOT BE CLOSED')
            reg_bounds.append(np.array(bounds))  # compile boundary defined

        # Bounds based on regions
        ws['region'] = ws['region'].astype('int')
        reg_bounds_val = [P[i] for i in reg_bounds]

        shp.update({
            'ws': ws,
            'reg_bounds': reg_bounds,
            'reg_bounds_val': reg_bounds_val,
            'no_reg': len(reg_bounds),
        })
        return shp

    def alphashape_3D(self, P, alpha):
        """
        Calculate the alphashape boundary from a point cloud P (3D np array)
        Compute the alpha shape (concave hull) of a set of 3D points.
        """
        from scipy.spatial import Delaunay
        import numpy as np
        import pandas as pd

        # Normalize points P
        normP = self.normP
        P = self.mm_norm(P, normP)

        spreadsheet = self.DTspreadsheet
        if type(spreadsheet) == type(None):
            # Get Delaunay triangulation of the points
            tetra = Delaunay(P)
            simps = tetra.simplices
            spreadsheet = pd.DataFrame(simps, columns = ['p1', 'p2', 'p3', 'p4'])
            tetraP = P[simps]

            # --- Calculating circumsphere radius and center (vectorised) --- #
            normsq = np.sum(tetraP**2, axis=2)[:, :, None]
            ones = np.ones((tetraP.shape[0], tetraP.shape[1], 1))

            A = np.linalg.det(np.concatenate((tetraP, ones), axis=2))
            A[A == 0] = 1e-30
            Dx = np.linalg.det(np.concatenate((normsq, tetraP[:,:,[1,2]], ones), axis = 2))
            Dy = -np.linalg.det(np.concatenate((normsq, tetraP[:,:,[0,2]], ones), axis = 2))
            Dz = np.linalg.det(np.concatenate((normsq, tetraP[:,:,[0,1]], ones), axis = 2))
            C = np.linalg.det(np.concatenate((normsq, tetraP), axis = 2))

            Ux = Dx/(2*A)
            Uy = Dy/(2*A)
            Uz = Dz/(2*A)
            U = np.array([Ux, Uy, Uz]).T   # center of circumsphere
            # Implement tolerance to prevent negative values
            sqrt_term = Dx**2 + Dy**2 + Dz**2 - 4*A*C
            spreadsheet['sqrt_term'] = sqrt_term
            sqrt_term[(sqrt_term < 0)*(sqrt_term > -1e20)] = 0
            if sqrt_term[sqrt_term == 0].shape[0] > 0:
                print('WARNING: Zero-volume tetrahedrons detected. These tetrahedrons will be ignored in point-in-polygon checks.')
            R = np.sqrt(sqrt_term)/(2*np.abs(A)) # radius of circumsphere
            
            spreadsheet[['Ux', 'Uy', 'Uz']] = U
            spreadsheet['r'] = R

            self.DTspreadsheet = spreadsheet
            
        alpha_spreadsheet = spreadsheet[spreadsheet['r'] <= alpha].copy()

        # ----- Get edge triangles only ----- #
        simps = alpha_spreadsheet[['p1', 'p2', 'p3', 'p4']].to_numpy()
        edgeComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]) # comb to separate tetrahedrons to triangles
        edges = simps[:, edgeComb].reshape(-1, 3) # separate tetra to tri
        edges.sort(axis = 1) # sort vertex indices
        edges = pd.DataFrame(edges).drop_duplicates(keep = False).to_numpy() # get unique entries only

        shp = {
            'P': self.mm_rev(P, normP),
            'simplices': simps,
            'edges': edges,
            'alpha': alpha,
            'alpha_spreadsheet': alpha_spreadsheet,
            'normalized_P': P,
            }
        return shp

    def categorise3D(self, ws, cur_idx, reg):
        """
        Breadth-first queue function to categorise triangles into regions
        ws: worksheet is a pandas dataframe containing all info
        cur_idx: current index in ws of the tri being checked (int)
        reg: region number (int)
        returns next index to be put in queue and mutates the original ws dataframe
        """
        import pandas as pd
        ws.loc[cur_idx, 'visit'] = True # set it to be visited
        ws.loc[cur_idx, 'region'] = reg # categorise to region reg

        # taking all directions to search neighbouring vertices
        current_pair1 = ws.loc[cur_idx]['p1']
        current_pair2 = ws.loc[cur_idx]['p2']
        current_pair3 = ws.loc[cur_idx]['p3']
        
        # make working_sheet for easier manipulation
        w_s = ws[ws['visit'] == False].copy()

        v1 = pd.concat([w_s[w_s['p1'] == current_pair1], 
                        w_s[w_s['p2'] == current_pair1], 
                        w_s[w_s['p3'] == current_pair1]])[['p1', 'p2', 'p3']]
        v2 = pd.concat([w_s[w_s['p1'] == current_pair2], 
                        w_s[w_s['p2'] == current_pair2], 
                        w_s[w_s['p3'] == current_pair2]])[['p1', 'p2', 'p3']]
        v3 = pd.concat([w_s[w_s['p1'] == current_pair3], 
                        w_s[w_s['p2'] == current_pair3], 
                        w_s[w_s['p3'] == current_pair3]])[['p1', 'p2', 'p3']]

        next_in_queue = []
        if v1.shape[0] != 0:
            next_in_queue.append(v1.index[0])
        if v2.shape[0] != 0:
            next_in_queue.append(v2.index[0])
        if v3.shape[0] != 0:
            next_in_queue.append(v3.index[0])
        return next_in_queue

    def classify_regions3D(self, shp = None):
        """
        Classify number of regions and their outer triangle surfaces
        """
        import numpy as np
        import pandas as pd

        # Unpack shp
        if type(shp) == None:
            shp = self.shp
        edges = shp['edges']
        P = shp['P']

        # ----- Identification of Regions ----- #
        # Implemented 3D breadth-first queue search categorisation
        sorted_tri = edges[edges[:, 0].argsort()]
        triComb = np.array([(0, 1), (0, 2), (1, 2)]) # comb to separate tri to lines

        # Create worksheet
        ws = pd.DataFrame(sorted_tri)
        ws['visit'] = False
        ws['region'] = np.nan

        # Connected triangles must share at least one pair of vertices
        ws['p1'] = [str(i) for i in sorted_tri[:, triComb][:, 0, :]]
        ws['p2'] = [str(i) for i in sorted_tri[:, triComb][:, 1, :]]
        ws['p3'] = [str(i) for i in sorted_tri[:, triComb][:, 2, :]]

        reg = 0
        # Loop over every triangle to categorise them into regions
        while ws[ws['visit'] == False].shape[0] > 0:
            reg += 1

            # find a point which have not been visited yet
            cur_idx = ws[ws['visit'] == False].index[0]
            history = []          # reset history
            queue = []            # reset the queue
            queue.append(cur_idx) # add the point into the queue
            
            # start queue for region reg
            while len(queue) != 0:
                cur_idx = queue.pop(0)
                history.append(cur_idx)
                next_in_queue = self.categorise3D(ws, cur_idx, reg)

                # Add next_in_queue into the queue and check for dupes
                # Also check if it has been covered in history
                queue += next_in_queue
                order = np.unique(queue, return_index=True)[1]
                queue = [queue[i] for i in sorted(order)]
                for i, j in enumerate(queue):
                    if j in history:
                        queue.pop(i)

        # Bounds based on regions
        ws['region'].astype('int')
        reg_bounds = [ws[ws['region'] == (i + 1)][[0, 1, 2]].to_numpy() for i in range(int(ws['region'].max()))]
        reg_bounds_val = [P[i] for i in reg_bounds]

        shp.update(
            {
            'ws': ws,
            'reg_bounds': reg_bounds, 
            'reg_bounds_val': reg_bounds_val,
            'no_reg': len(reg_bounds),
            }
        )
        return shp

    def inside2D(self, x, shp = None):
        """
        Point in triangle checks based on Barycentric coordinates vectorised
        Brute force check on all simplices in shp
        Splits option included to handle memory limitations
        Returns False if x is not in self.shp, True otherwise (2D)
        x: list of [x, y], or [[x1, y1], [x2, y2], ...]
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
                x_list = np.array_split(x, no_splits)
        else:
            x_list = [np.array(x)]

        # Find which triangle the point lies in
        P = shp['P']
        simps = shp['simplices']

        res_list = []
        bool_list = []
        for X in x_list:
            # Convert all vertices to Barycentric coordinates wrt v0
            v0 = P[simps[:, 0],:]
            v1 = P[simps[:, 1],:] - v0
            v2 = P[simps[:, 2],:] - v0

            # Compute transformation matrix mat
            n_tri = len(simps)
            v1r = v1.T.reshape((2, 1, n_tri))
            v2r = v2.T.reshape((2, 1, n_tri))
            mat = np.concatenate((v1r, v2r), axis = 1)

            # Get inverse of transformation matrix for each triangle
            inv_mat = np.linalg.inv(mat.T).T

            # Assemble vectors for vectorised calculation wrt length of X
            if X.size == 2:
                X = X.reshape((1,2))
            n_X = X.shape[0]
            v0p = np.repeat(v0[:,:,np.newaxis], n_X, axis = 2)

            # Transform X based on the origin of each local tetrahedral coordinate system
            transX = np.einsum('imk,kmj->kij', inv_mat, X.T - v0p)
            
            # Perform point check:
            #     All transformed coordinates has to be between 0 and 1
            #     and the sum of the coordinates is less than or equal to 1
            val = np.all(transX >=0, axis = 1) & np.all(transX <= 1, axis = 1) & (np.sum(transX, axis = 1) <= 1)
            id_tet, id_X = np.nonzero(val) # get indices of satisfied conditions
            res = -np.ones(n_X, dtype = id_tet.dtype) # setup results array
                                                        # assuming -1 initially for all entries
            res[id_X] = id_tet # if the point lies inside, then replace the -1 values with id_tet
            
            # Recording
            res_list.append(res)
            bool_list.append(res != -1)
        res = np.concatenate(res_list)
        bool_res = np.concatenate(bool_list)

        if dim == 1:
            return bool_res[0]
        return bool_res
    
    def inside3D(self, x, shp = None):
        """
        Point in tetrahedron checks based on Barycentric coordinates vectorised
        Brute force check on all simplices in shp
        Splits option included to handle memory limitations
        Returns False if x is not in self.shp, True otherwise (3D)
        x: list of [x, y, z], or [[x1, y1, z1], [x2, y2, z2], ...]
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
                x_list = np.array_split(x, no_splits)
        else:
            x_list = [np.array(x)]

        # Find which tetrahedron the point lies in
        P = shp['P']
        simps = shp['simplices']

        res_list = []
        bool_list = []
        for X in x_list:
            # Convert all vertices to Barycentric coordinates wrt v0
            v0 = P[simps[:, 0],:]
            v1 = P[simps[:, 1],:] - v0
            v2 = P[simps[:, 2],:] - v0
            v3 = P[simps[:, 3],:] - v0

            # Compute transformation matrix mat
            n_tet = len(simps)
            v1r = v1.T.reshape((3, 1, n_tet))
            v2r = v2.T.reshape((3, 1, n_tet))
            v3r = v3.T.reshape((3, 1, n_tet))
            mat = np.concatenate((v1r, v2r, v3r), axis = 1)

            # Get inverse of transformation matrix for each tetrahedron
            non_singular_index = np.linalg.det(mat.T) != 0
            non_singular_mat = mat.T[non_singular_index]
            # if sum(non_singular_index) < mat.T.shape[0]:
                # print('WARNING: Singular matrices detected and will be ignored in point-in-polygon checks.')
            inv_mat = np.linalg.inv(non_singular_mat).T

            # Assemble vectors for vectorised calculation wrt length of X
            if X.size == 3:
                X = X.reshape((1, 3))
            n_X = X.shape[0]
            v0p = np.repeat(v0[:,:,np.newaxis], n_X, axis = 2)

            # Transform X based on the origin of each local tetrahedral coordinate system
            transX = np.einsum('imk,kmj->kij', inv_mat, (X.T - v0p)[non_singular_index])
            
            # Perform point check:
            #     All transformed coordinates has to be between 0 and 1
            #     and the sum of the coordinates is less than or equal to 1
            val = np.all(transX >=0, axis = 1) & np.all(transX <= 1, axis = 1) & (np.sum(transX, axis = 1) <= 1)
            id_tet, id_X = np.nonzero(val) # get indices of satisfied conditions
            res = -np.ones(n_X, dtype = id_tet.dtype) # setup results array
                                                       # assuming -1 initially for all entries
            res[id_X] = id_tet # if the point lies inside, then replace the -1 values with id_tet
            
            # Recording
            res_list.append(res)
            bool_list.append(res != -1)
        res = np.concatenate(res_list)
        bool_res = np.concatenate(bool_list)

        if dim == 1:
            return bool_res[0]
        return bool_res

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
            fs_df = None
            fs_R = {'x': x, 'FR': 'N/A', 'rmax': 'N/A', 'rmin': 'N/A',
            'space_size': 'N/A', 'plusmin': 'N/A', 'nosam': 'N/A', 
            'hmv': 'N/A', 'hmv_sam_flag': 'N/A', 'not_in_region_flag': not_in_region_flag, 'fs_df': fs_df}
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
            fs_size = self.mm_size_norm((rmax - rmin).prod(), self.normP)
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
        
        return None
    
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
        f.write(f'Number of regions: {rp["no_reg"]:10d}\n')
        f.write(f'Number of samples in DSp: {self.sat.shape[0]}\n\n')
        f.write(f'Average {rp["hmv"]["name"]}:    {rp["hmv"]["mean"]:10}\n')
        f.write(f'DS Maximum {rp["hmv"]["name"]}: {rp["hmv"]["max"]:10}\n')
        f.write(f'DS Minimum {rp["hmv"]["name"]}: {rp["hmv"]["min"]:10}\n')
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
                                f'{rep["hmv"]["mean"]:10}\n')
                            f.write(f'DS Maximum {rep["hmv"]["name"]}:' +\
                                f'{rep["hmv"]["max"]:10}\n')
                            f.write(f'DS Minimum {rep["hmv"]["name"]}:' +\
                                f'{rep["hmv"]["min"]:10}\n')
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
        return None
    
def save_ds(ds, project_name):
    # Processing functions:
    def convert_shp(shp):
        import pandas as pd
        # Setup empty dataframe
        NAdf = pd.DataFrame(['not_available'])

        # convert shp entries to lists and extract the dataframes
        ws = shp.pop('ws', NAdf)
        alpha_spreadsheet = shp.pop('alpha_spreadsheet', NAdf)

        shp['P'] = shp['P'].tolist()
        shp['simplices'] = shp['simplices'].tolist()
        shp['edges'] = shp['edges'].tolist()
        shp['alpha'] = float(shp['alpha'])
        shp['reg_bounds'] = [i.tolist() for i in shp['reg_bounds']]
        shp['reg_bounds_val'] = [i.tolist() for i in shp['reg_bounds_val']]
        shp['size'] = float(shp['size'])
        if type(shp.get('normalized_P', None)) != type(None):
            shp['normalized_P'] = shp['normalized_P'].tolist()
        return shp, ws, alpha_spreadsheet

    def convert_report(report):
        ss = report['space_size']
        if type(ss) != str:
            report['space_size'] = float(ss)
        else:
            report['space_size'] = ss
        max_sample = report['hmv']['max_sample']
        if type(max_sample) != str:
            report['hmv']['max_sample'] = max_sample.to_dict()
        else:
            report['hmv']['max_sample'] = 'N/A'
        min_sample = report['hmv']['min_sample']
        if type(max_sample) != str:
            report['hmv']['min_sample'] = min_sample.to_dict()
        else:
            report['hmv']['min_sample'] = 'N/A'
        return report

    def convert_all_x(point_in_polygon_results):
        import numpy as np
        keys = list(point_in_polygon_results.keys())
        for k in keys:
            test_result = point_in_polygon_results[k]
            test_result['FR'] = np.array(test_result['FR']).tolist()
            test_result['rmax'] = np.array(test_result['rmax']).tolist()
            test_result['rmin'] = np.array(test_result['rmin']).tolist()
            if test_result['space_size'] == 'N/A':
                pass
            else:
                test_result['space_size'] = float(test_result['space_size'])
            test_result['plusmin'] = np.array(test_result['plusmin']).tolist()
            
            if test_result['hmv'] == 'N/A':
                pass
            else:
                max_sample = test_result['hmv']['max_sample']
                if type(max_sample) != str:
                    test_result['hmv']['max_sample'] = max_sample.to_dict()
                else:
                    test_result['hmv']['max_sample'] = 'N/A'
                min_sample = test_result['hmv']['min_sample']
                if type(min_sample) != str:
                    test_result['hmv']['min_sample'] = min_sample.to_dict()
                else:
                    test_result['hmv']['min_sample'] = 'N/A'
            
            opt_log = test_result.get('AORopt_log', 'N/A')

            if opt_log == 'N/A':
                test_result['AORopt_log'] = 'N/A'
            else:
                for i in range(len(opt_log)):
                    opt_log[i]['verts'] = opt_log[i]['verts'].tolist()
                test_result['AORopt_log'] = opt_log
            
            point_in_polygon_results[k] = test_result
        return point_in_polygon_results

    # Save design space project as a zip file containing many different files including the datasets
    # make a folder named 'DS' in the current directory
    import json
    import os
    import copy
    import pandas as pd
    tmp_project_name = f'__tmp__/{project_name}'
    os.makedirs(tmp_project_name, exist_ok = True)
    os.makedirs(tmp_project_name + '/csv', exist_ok = True)
    os.makedirs(tmp_project_name + '/shp', exist_ok = True)
    os.makedirs(tmp_project_name + '/point_in_polygon', exist_ok = True)

    saved_ds = copy.deepcopy(ds)
    if hasattr(saved_ds, 'vnames'):
        saved_ds.send_output(tmp_project_name + '/ds_report')
    # Setup empty dataframe
    NAdf = pd.DataFrame(['not_available'])

    # Save pandas dataframes as csv
    df = getattr(saved_ds, 'df', NAdf)
    sat = getattr(saved_ds, 'sat', NAdf)
    vio = getattr(saved_ds, 'vio', NAdf)
    vindsp = getattr(saved_ds, 'vindsp', NAdf)
    df.to_csv(tmp_project_name + '/csv/df.csv')
    sat.to_csv(tmp_project_name + '/csv/sat.csv')
    vio.to_csv(tmp_project_name + '/csv/vio.csv')
    vindsp.to_csv(tmp_project_name + '/csv/vindsp.csv')

    # Save options file
    dside_opt = getattr(saved_ds, 'opt', {'N/A': 'N/A'})
    extra_points = dside_opt.pop('extra_points')
    if type(extra_points) != type([]):
        pd.DataFrame(extra_points).to_csv(tmp_project_name + '/csv/extra_points.csv')
    else:
        NAdf.to_csv(tmp_project_name + '/csv/extra_points.csv')
    with open(tmp_project_name + '/dside_opt.json', 'w') as f:
        json.dump(dside_opt, f)

    # Saving constraints file
    constraints = getattr(saved_ds, 'constraints', 'no_constraints')
    with open(tmp_project_name + '/constraints.json', 'w') as f:
        json.dump(constraints, f)

    # Save shp file
    shp = getattr(saved_ds, 'shp', False)
    if shp == False:
        shp = {'N/A': 'N/A'}
    else:
        shp, ws, alpha_spreadsheet = convert_shp(shp)
        ws.to_csv(tmp_project_name + '/shp/ws.csv')
        alpha_spreadsheet.to_csv(tmp_project_name + '/shp/alpha_spreadsheet.csv')
    with open(tmp_project_name + '/shp/shp.json', 'w') as f:
        json.dump(shp, f)

    # Save report file
    report = getattr(saved_ds, 'report', {'N/A': 'N/A'})
    report = convert_report(report)
    with open(tmp_project_name + '/report.json', 'w') as f:
        json.dump(report, f)

    # Save point_in_polygon files
    point_in_polygon_results = saved_ds.all_x
    point_in_polygon_results = convert_all_x(point_in_polygon_results)
    keys = list(point_in_polygon_results.keys())
    point_in_polygon_report = {'no_of_checks': len(keys)}
    with open(tmp_project_name + '/point_in_polygon/point_in_polygon_report.json', 'w') as f:
        json.dump(point_in_polygon_report, f)
    for i in range(len(keys)):
        os.makedirs(tmp_project_name + f'/point_in_polygon/{i+1:06d}', exist_ok = True)
    folders = os.listdir(tmp_project_name + '/point_in_polygon')
    for i, k in enumerate(keys):
        test_result = point_in_polygon_results[k]
        fs_df = test_result.pop('fs_df')
        if type(fs_df) == type(None):
            fs_df = NAdf
        fs_df.to_csv(tmp_project_name + f'/point_in_polygon/{folders[i]}/fs_df.csv')
        
        if test_result['hmv'] == 'N/A':
            NAdf.to_csv(tmp_project_name + f'/point_in_polygon/{folders[i]}/fs_all_samples.csv')
        else:
            if type(test_result['hmv']['fs_all_samples']) != str:
                fs_all_samples = test_result['hmv'].pop('fs_all_samples')
                fs_all_samples.to_csv(tmp_project_name + f'/point_in_polygon/{folders[i]}/fs_all_samples.csv')
            else:
                NAdf.to_csv(tmp_project_name + f'/point_in_polygon/{folders[i]}/fs_all_samples.csv')
        with open(tmp_project_name + f'/point_in_polygon/{folders[i]}/test_result.json', 'w') as f:
            json.dump(test_result, f)

    # Zip up the project folder
    import zipfile
    extension_name = '.myDSp'
    with zipfile.ZipFile(project_name + extension_name, 'w') as zipf:
        for root, dirs, files in os.walk(tmp_project_name):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), tmp_project_name))

    # Clean up the project folder
    import shutil
    shutil.rmtree('__tmp__', ignore_errors = True)
    return None


def load_ds(project_name):
    import os
    import numpy as np
    import pandas as pd
    import json
    from dside import DSI

    # Loading the DSp project
    tmp_folder = '__tmp__'
    project_folder = f'{tmp_folder}/{project_name}'
    os.makedirs(tmp_folder, exist_ok = True)

    # Unzip the project file
    import zipfile
    with zipfile.ZipFile(project_name, 'r') as zip_ref:
        zip_ref.extractall(project_folder)

    NAdf = pd.DataFrame(['not_available'])

    # Creating DSI instance
    df = pd.read_csv(project_folder + '/csv/df.csv', index_col = 0)
    ds = DSI(df)

    # Loading data files
    sat = pd.read_csv(project_folder + '/csv/sat.csv', index_col = 0)
    vio = pd.read_csv(project_folder + '/csv/vio.csv', index_col = 0)
    vindsp = pd.read_csv(project_folder + '/csv/vindsp.csv', index_col = 0)
    if (sat.to_numpy() == NAdf.to_numpy()).all():
        pass
    else:
        ds.sat = sat
    if (vio.to_numpy() == NAdf.to_numpy()).all():
        pass
    else:
        ds.vio = vio
    if (vindsp.to_numpy() == NAdf.to_numpy()).all():
        pass
    else:
        ds.vindsp = vindsp

    # dside options loading
    with open(project_folder + '/dside_opt.json', 'r') as f:
        ds.opt = json.load(f)
    ds.opt['extra_points'] = pd.read_csv(project_folder + '/csv/extra_points.csv', index_col = 0).to_numpy()
    if ds.opt['extra_points'].shape == (1, 1):
        ds.opt['extra_points'] = []
    if ds.opt.get('vnames', None) != None:
        ds.vnames = ds.opt['vnames']

    # Loading constraints file
    with open(project_folder + '/constraints.json', 'r') as f:
        constraints = json.load(f)
    if type(constraints) == type({}):
        ds.constraints = constraints

    # Converting back the shp files
    with open(project_folder + '/shp/shp.json', 'r') as f:
        shp = json.load(f)
    if shp.get('N/A', False) == 'N/A':
        shp = None
    else:
        shp['P'] = np.array(shp['P'])
        shp['simplices'] = np.array(shp['simplices'])
        shp['edges'] = np.array(shp['edges'])
        shp['reg_bounds'] = [np.array(i) for i in shp['reg_bounds']]
        shp['reg_bounds_val'] = [np.array(i) for i in shp['reg_bounds_val']]
        if type(shp.get('normalized_P', None)) != type(None):
            shp['normalized_P'] = np.array(shp['normalized_P'])
        shp['ws'] = pd.read_csv(project_folder + '/shp/ws.csv', index_col = 0)
        shp['alpha_spreadsheet'] = pd.read_csv(project_folder + '/shp/alpha_spreadsheet.csv', index_col = 0)
        ds.shp = shp

    # Setting ds.space_size
    if hasattr(ds, 'shp'):
        ds.space_size = ds.shp['size']
    else:
        ds.space_size = None

    # Loading report files
    with open(project_folder + '/report.json', 'r') as f:
        report = json.load(f)
    max_sample = report['hmv']['max_sample']
    if type(max_sample) == type({}):
        max_sample = pd.DataFrame(max_sample)
    else:
        max_sample = '-'
    min_sample = report['hmv']['min_sample']
    if type(min_sample) == type({}):
        min_sample = pd.DataFrame(min_sample)
    else:
        min_sample = '-'
    report['hmv']['max_sample'] = max_sample
    report['hmv']['min_sample'] = min_sample
    ds.report = report

    # Loading point_in_polygon files
    test_folders = [f for f in os.listdir(project_folder + '/point_in_polygon') if os.path.isdir(os.path.join(project_folder + '/point_in_polygon', f))]
    all_x = {}
    if len(test_folders) == 0:
        pass
    else:
        for i, folder in enumerate(test_folders):
            with open(project_folder + f'/point_in_polygon/{folder}/test_result.json', 'r') as f:
                test_result = json.load(f)
            k = str(test_result['x'])

            if type(test_result['FR']) != type(''):
                test_result['FR'] = np.array(test_result['FR'])
            if type(test_result['rmax']) != type(''):
                test_result['rmax'] = np.array(test_result['rmax'])
            if type(test_result['rmin']) != type(''):
                test_result['rmin'] = np.array(test_result['rmin'])
            if type(test_result['plusmin']) != type(''):
                test_result['plusmin'] = np.array(test_result['plusmin'])
            
            fs_df = pd.read_csv(project_folder + f'/point_in_polygon/{folder}/fs_df.csv', index_col = 0)
            if (fs_df.to_numpy() == NAdf.to_numpy()).all():
                fs_df = None
            test_result['fs_df'] = fs_df

            if test_result['hmv'] == 'N/A':
                pass
            else:
                max_sample = test_result['hmv']['max_sample']
                if type(max_sample) != str:
                    test_result['hmv']['max_sample'] = pd.DataFrame(max_sample)
                min_sample = test_result['hmv']['min_sample']
                if type(min_sample) != str:
                    test_result['hmv']['min_sample'] = pd.DataFrame(min_sample)
                fs_all_samples = pd.read_csv(project_folder + f'/point_in_polygon/{folder}/fs_all_samples.csv', index_col = 0)
                if (fs_all_samples.to_numpy() == NAdf.to_numpy()).all():
                    fs_all_samples = '-'
                test_result['hmv']['fs_all_samples'] = fs_all_samples
                
            
            
            opt_log = test_result.get('AORopt_log', 'N/A')
            if opt_log == 'N/A':
                pass
            else:
                for i in range(len(opt_log)):
                    opt_log[i]['verts'] = np.array(opt_log[i]['verts'])
                test_result['AORopt_log'] = opt_log
            all_x[k] = test_result
    ds.all_x = all_x

    import shutil
    shutil.rmtree('__tmp__')
    return ds