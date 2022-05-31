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
            'czorder': False,         # computed_zorder for 3D plots settings
            'fs': (6, 4),            # Figure size
            'bw': False,             # If True, use black-white template
            'alpha': 0.45,           # Transparency of points
            'font_size': 10,         # Font size
            # Color bar and map
            'cbarloc': 'right',   # Colobar location
            'cbaror': 'vertical', # Colorbar orientation
            'mycmap': None,       # Use your own cmap (input name of cmap as str)
            # Satisfied samples
            'satlabel': 'Satisfied', # Satisfied samples label
            'satcolor': 'g',         # Satisfied samples color
            'satmarker': 'o',        # Marker of satisfied points
            'satfill': 'g',          # Marker fill color of satisfied points
            'satzorder': 5,          # Decides which level to be plotted on
            # Violated samples
            'violabel': 'Violated',  # Violated samples label
            'viocolor': 'r',         # Violated samples color
            'viomarker': 'o',        # Marker of violated points
            'viofill': 'r',          # Marker fill color of violated points
            'viozorder': 5,          # Decides which level to be plotted on
            # Legend
            'legloc': 'best',        # Legend location
            'framealpha': 0.8,       # Legend box transparency
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
            'dspwidth': 3,       # Thickness of the boundary (2D)
            'dspstyle': '-',     # Line style of the boundary (2D)
            'dspalpha': 0.2,     # Transparency of surface/boundary (3D)
            'dspzorder': 20,     # To make sure it is plotted ontop of the samples
            
            # ----- NOP Parameters ----- #
            'step_change': 1,    # Step change of expanding AOR in percent
            'noplabel': 'NOP',   # Normal Operating Point label for legend
            'nopmarker': 'x',    # Nominal operating point marker style
            'nopwidth':   3,     # Nominal operating point marker thickness
            'nopcolor': 'black', # Nominal operating point marker color
            'nopsize':  100,     # Nominal operating point marker size
            'nopzorder': 10,     # To make sure it is plotted ontop of the samples

            # ----- AOR Parameters ----- #
            'aorlabel': 'AOR',   # Uniform Proven Acceptable Range
            'aorstyle': '--',    # AOR boundary line style
            'aorcolor': 'black', # AOR boundary line color
            'aorwidth': 3,       # AOR boundary line width
            'aorzorder': 10,     # To make sure it is plotted ontop of the samples
            
            # ----- Hull Parameters ----- #
            'a': None, # Alpha value -> at large alpha,
                       # hull becomes convex. if set to 'critical', MATLAB finds 
                       # the smallest one. if None: use mean of bounds of dimensions
            'amul': 1, # Alpha multiplier value (wrt to product of mean axes)
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
        data = self.df
        self.constraints = constraints
        sat = data.copy()
        for i in list(constraints.keys()):
            sat = sat[sat[i] >= constraints[i][0]]
            sat = sat[sat[i] <= constraints[i][1]]
        exclude_these = data.index.isin(list(sat.index))
        vio = data[~exclude_these]
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
    
    def plot(self, vnames, opt = {}):
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
                        location = opt['cbarloc'], orientation = opt['cbaror'])
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
                self.space_size, bF, P, self.shp = self.find_DSp(vnames, opt)
            bF = self.bF
            P = self.P
            if dim == 2:
                for i in range(bF.shape[0]):
                    if i == 0:
                        plt.plot(P[bF][i][:, 0], P[bF][i][:, 1],\
                            linewidth = opt['dspwidth'], linestyle = opt['dspstyle'],\
                            color = opt['dspcolor'], label = opt['dsplabel'],\
                            zorder = opt['dspzorder'])
                    else:
                        plt.plot(P[bF][i][:, 0], P[bF][i][:, 1],\
                            linewidth = opt['dspwidth'], linestyle = opt['dspstyle'],\
                            color = opt['dspcolor'], zorder = opt['dspzorder'])

            if dim == 3:
                surf = ax.plot_trisurf(*zip(*P), triangles = bF, color = opt['dspcolor'],\
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
    
    def find_DSp(self, vnames, opt = {}):
        """
        Create a hull using MATLAB alphashape
        Depending on alpha (opt['a']) value it can be either convex/concave
        # ----- Hull Parameters ----- #
        'a': None, # Alpha value -> at large alpha, hull becomes convex.
        If set to 'critical', MATLAB finds the smallest one. if None: use mean of bounds 
        of dimensions
        'amul': 0.5, # Alpha multiplier value (wrt to product of mean axes)
        """
        import matlab.engine
        import numpy as np
        import pandas as pd
        eng = matlab.engine.start_matlab() # Start instance of matlab engine
        ev = eng.eval
        ew = eng.workspace
        
        sat = self.sat
        vio = self.sat
        self.opt.update({'vnames': vnames})
        self.opt.update(opt) 
        opt = self.opt
        a = opt['a']
        amul = opt['amul']
        points = sat[vnames].to_numpy()
        
        ew['points'] = matlab.double(points.tolist()) 
        if a == 'critical':
            shp = ev(f"alphaShape(points)", nargout = 1)
        elif a == None:
            a = pd.concat([sat, vio], axis = 0)[vnames].mean().product()*amul
            shp = ev(f"alphaShape(points, {a})", nargout = 1)
        else:
            shp = ev(f"alphaShape(points, {a})", nargout = 1)
        ew['shp'] = shp
        
        if points.shape[1] == 2:
            space_size = eng.area(shp)
        elif points.shape[1] == 3:
            space_size = eng.volume(shp)
        
        bF, P = eng.boundaryFacets(shp, nargout = 2)
        bF = np.array(bF).astype('int') - 1
        P = np.array(P)
        
        self.shp = shp
        self.space_size = space_size
        self.bF = bF
        self.P = P
        self.report['space_size'] = space_size
        return space_size, bF, P, shp
    
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
    
    def check_point(self, x):
        """
        Checks whether point x lies within the hull or not.
        """
        import numpy as np
        import matlab.engine
        eng = matlab.engine.start_matlab() # Start instance of matlab engine
        
        shp = self.shp
        vnames = self.vnames
        opt = self.opt
        sat = self.sat
        
        dim = len(vnames)
        step_change = opt['step_change']
        df = self.df
                        
        # Finding space boundary
        inputs_max = df[vnames].max().to_numpy()
        inputs_min = df[vnames].min().to_numpy()
        inputs_range = inputs_max - inputs_min
        pc = step_change/100
        
        not_in_region_flag = False
        if eng.inShape(shp, matlab.double(list(x))) == False:
            print('x is not inside DSp.')
            not_in_region_flag = True
            fs_R = {'x': x, 'FR': 'N/A', 'rmax': 'N/A', 'rmin': 'N/A',
            'space_size': 'N/A', 'plusmin': 'N/A', 'nosam': 'N/A', 
            'hmv': 'N/A', 'hmv_sam_flag': 'N/A', 'not_in_region_flag': not_in_region_flag}
        else:
            # ----- 2D space ----- #
            if dim == 2:
                flag = False
                while flag == False: # Going outwards
                    fsi   = np.array(
                        [[x[0] - pc*inputs_range[0], x[1] - pc*inputs_range[1]],
                        [x[0] - pc*inputs_range[0], x[1] + pc*inputs_range[1]],
                        [x[0] + pc*inputs_range[0], x[1] + pc*inputs_range[1]],
                        [x[0] + pc*inputs_range[0], x[1] - pc*inputs_range[1]]]
                        )
                    flag = False in [eng.inShape(shp, matlab.double(list(fsi[i])))\
                        for i in range(fsi.shape[0])]
                    pc += step_change/100

                pc = (step_change/100)/100
                flag = True
                while flag == True: # Going inwards
                    fs = [
                        [fsi[0, 0] + pc*inputs_range[0], fsi[0, 1] + pc*inputs_range[1]],
                        [fsi[1, 0] + pc*inputs_range[0], fsi[1, 1] - pc*inputs_range[1]],
                        [fsi[2, 0] - pc*inputs_range[0], fsi[2, 1] - pc*inputs_range[1]],
                        [fsi[3, 0] - pc*inputs_range[0], fsi[3, 1] + pc*inputs_range[1]]
                        ]
                    fs = np.array(fs)
                    flag = False in [eng.inShape(shp, matlab.double(list(fs[i])))\
                        for i in range(fs.shape[0])]
                    pc += (step_change/100)/100
                    
                fs = fs.tolist()
                fs.append(fs[0])
                fs_arr = np.array(fs)
                FR = np.array(fs)
                
            # ----- 3D space ----- #
            if dim == 3: 
                flag = False
                while flag == False: # Going outwards
                    fsi = np.array(
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
                    flag = False in [eng.inShape(shp, matlab.double(list(fsi[i])))\
                        for i in range(fsi.shape[0])]
                    pc += step_change/100

                pc = (step_change/100)/100
                flag = True
                while flag == True: # Going inwards
                    fs = [[fsi[0, 0] + pc*inputs_range[0],\
                         fsi[0, 1] + pc*inputs_range[1], fsi[0, 2] + pc*inputs_range[2]],
                        [fsi[1, 0] - pc*inputs_range[0],\
                        fsi[1, 1] + pc*inputs_range[1], fsi[1, 2] + pc*inputs_range[2]],
                        [fsi[2, 0] + pc*inputs_range[0],\
                        fsi[2, 1] - pc*inputs_range[1], fsi[2, 2] + pc*inputs_range[2]],
                        [fsi[3, 0] + pc*inputs_range[0],\
                        fsi[3, 1] + pc*inputs_range[1], fsi[3, 2] - pc*inputs_range[2]],
                        [fsi[4, 0] - pc*inputs_range[0],\
                        fsi[4, 1] - pc*inputs_range[1], fsi[4, 2] + pc*inputs_range[2]],
                        [fsi[5, 0] - pc*inputs_range[0],\
                        fsi[5, 1] - pc*inputs_range[1], fsi[5, 2] - pc*inputs_range[2]],
                        [fsi[6, 0] - pc*inputs_range[0],\
                        fsi[6, 1] + pc*inputs_range[1], fsi[6, 2] - pc*inputs_range[2]],
                        [fsi[7, 0] + pc*inputs_range[0],\
                        fsi[7, 1] - pc*inputs_range[1], fsi[7, 2] - pc*inputs_range[2]]]
                    fs = np.array(fs)
                    fs_arr = fs
                    flag = False in [eng.inShape(shp, matlab.double(list(fs[i])))\
                        for i in range(fs.shape[0])]
                    pc += (step_change/100)/100
                    
                FR = [[fs[0], fs[1], fs[4], fs[2], fs[0]],
                        [fs[0], fs[3], fs[6], fs[1], fs[0]],
                        [fs[3], fs[7], fs[5], fs[6], fs[3]], 
                        [fs[6], fs[1], fs[4], fs[5], fs[6]],
                        [fs[7], fs[3], fs[0], fs[2], fs[7]],
                        [fs[2], fs[4], fs[5], fs[7], fs[2]]]

            # ----- KPIs ----- #
            rmax = fs_arr.max(axis = 0)
            rmin = fs_arr.min(axis = 0)
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
                    fs_df = fs_df[fs_df[vnames[i]] <= rmax[i]]
                for i in range(dim):
                    fs_df = fs_df[fs_df[vnames[i]] >= rmin[i]]
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
                'hmv_sam_flag': no_samples_flag, 'not_in_region_flag': not_in_region_flag}
                
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
    
    def send_output(self, output_filename = 'DSI_output', appendix = True,\
            rp_pkl = True):
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