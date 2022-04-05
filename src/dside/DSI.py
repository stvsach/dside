from lib2to3.pgen2.pgen import DFAState

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
    constraints: dictionary containing name of variable and list of [lower bound, upper bound]
    vnames: list of manipulated variable names
    x: list of nominal point for flexible space analysis
    opt: options for the lotting
    """
    from dside import DSI
    
    ds = DSI(df)
    p = ds.screen(constraints)
    r = ds.plot(vnames, opt)
    if x != None:
        r = ds.flex_space(x)
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
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        # Set internal definition
        self.df = df
        self.report = {
            'nosam': df.shape[0], 'space_size': '-',
            'hmv': {'name': 'None', 'mean': '-', 'max': '-', 'max_sample': '-', 'min': '-',  'min_sample': '-'},
        }
        self.all_x = []
        
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
            'xlabel': 'x',     # x axis label
            'ylabel': 'y',     # y axis label
            'zlabel': 'z',     # z axis label
            'cmap': 'inferno80',
            'hmv': 'None', # heat map variable name
            'hmvlabel': 'hmvlabel: heat map var label', # heat map variable label
            'nplabel': 'NP',  # x label for flex space
            'fslabel': 'FR',  # flexible region label
            'spacelabel': 'NOR', # Label of surface/boundary
            
            # ----- Hidden Elements ----- #
            'hidehmv': False, # If True, no heat map will be plotted
            'hidesat': False, # If True, no satisfied variables will be plotted
            'hidevio': False, # If True, no violated variables will be plotted
            'hidenor': False, # If True, hides the surface/boundary
            
            # ----- Plot Format ----- #
            'fs': (6, 4),        # Figure size
            'bw': False,         # If True, use black-white template
            'csat': 'g',         # Satisfied samples color
            'cvio': 'r',         # Violated samples color
            'msat': 'o',         # Marker of satisfied points
            'mvio': 'o',         # Marker of violated points
            'fsat': 'g',         # Marker fill color of satisfied points
            'fvio': 'r',         # Marker fill color of violated points
            'lsat': 'Satisfied', # Satisfied samples label
            'lvio': 'Violated',  # Violated samples label
            'alpha': 0.45,       # Transparency of points
            'legloc': 'best',    # Legend location
            'font_size': 10,     # Font size
            'elev': 20,          # Elevation of 3D plot
            'azim': -70,         # Azimuth of 3D plot
            'limfactor': 0.05,   # Axes limit factor based on range of axes
            'axeslimdf': 'df',   # Data used to calculate axes limits ('sat', 'vio', 'df', or 'best')
            
            # ----- Space Format ----- #
            'cspace': 'black',                # Color of the surface/boundary
            'alphaspace': 0.2,            # Transparency of surface/boundary
            
            # ----- Flex Space Parameters ----- #
            'step_change': 1,   # Step change of expanding flex space in percent
            'npmarker': 'x',    # Nominal point marker
            'npcolor': 'black', # Nominal point color
            'fsstyle': '--',     # Boundary line style
            'fscolor': 'black', # Boundary line color
            
            # ----- Convex Hull Parameters ----- #
            'a': None,           # Alpha value -> at large alpha hull becomes convex. if set to None, MATLAB finds the smallest one
            'amul': 1,           # Alpha multiplier value (wrt to product of mean axes)
        }
        self.opt = self.default_opt.copy()
        return None
    
    def reset(self):
        """
        Reset options to default.
        """
        self.opt = self.default_opt.copy()
        return None
    
    def screen(self, constraints):
        """
        Takes in the DataFrame, data, and dictionary, constraints, giving out the satisfied and violated DataFrame of samples
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
    
    def help(self):
        """
        Prints usage instructions and ALL of the current options and return the opt dictionary.
        """
        print('# ----- Usage Instructions ----- #')
        print('1. ds = dside.DSI(df)         # Create instance of design space ds with data from DataFrame df')
        print('2. p = ds.screen(constraints) # Screen the points using the constraints (dictionary)')
        print('3. r = ds.plot(vnames)        # Plot the design space and NOR based on vnames (list of variable names for the axes)')
        print('4. r = ds.flex_space(x)       # Plot the nominal point and flexibility region based on point x (list/numpy array)')
        print('\n# ----- Options ----- #')
        for i in list(self.opt.keys()):
            print(f'{i:10}: {self.opt[i]}')
        return self.opt
    
    def plot(self, vnames, opt = {}):
        """
        Plot either 2D or 3D design space based on satisfied and violated points.
        vnames: ['varname1', 'varname2', 'varname3']
        Plotting options available by passing opt which is a dictionary to update the default self.opt.
        """
        
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        # ------------------------------ Internal definitions ------------------------------ #
        self.vnames = vnames
        axes_label = ['xlabel', 'ylabel', 'zlabel']
        for i, l in enumerate(vnames):
            self.opt[axes_label[i]] = l
        self.opt.update(opt)
        if self.opt['bw']:
            bw_template = {'csat': 'gray', 'cvio': 'gray', 'fsat': 'gray', 'fvio': 'white', 'cmap': 'gray80',
                           'msat': 'o', 'mvio': 's'}
            self.opt.update(bw_template)
        self.opt.update(opt)
        
        # ------------------------------ External definitions ------------------------------ #
        sat = self.sat
        vio = self.vio
        opt = self.opt    
        plt.rcParams.update({'font.size': opt['font_size']})
        
        # ------------------------------ Heat map Variable ------------------------------ #
        hmv_R = {'name': opt['hmv'], 'mean': '-', 'max': '-', 'max_sample': '-', 'min': '-',  'min_sample': '-'}
        if opt['hmv'] != 'None':
            if opt['hidesat']:
                hmvdf = vio.copy()
            elif opt['hidevio']:
                hmvdf = sat.copy()
            else:
                hmvdf = pd.concat([sat, vio], axis = 0)
            hmvdf = hmvdf[opt['hmv']]
            norm = matplotlib.colors.Normalize(vmin = hmvdf.min(), vmax = hmvdf.max())
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
            ax = fig.add_subplot(projection = '3d')
        elif dim > 3:
            print('Dimension of vnames is larger than 3.')
        self.ax = ax
        
        # Scatter plot        
        if opt['hidesat'] == False:
            if nosat_flag == False:
                if (opt['hmv'] == 'None') or (opt['hidehmv'] == True):
                    ax.scatter(*zip(*sat[vnames].to_numpy()), marker = opt['msat'], facecolors = opt['fsat'], label = opt['lsat'], color = opt['csat'], alpha = opt['alpha'])
                else:
                    fig.colorbar(sm, label = opt['hmvlabel'])
                    ax.scatter(*zip(*sat[vnames].to_numpy()), marker = opt['msat'], label = opt['lsat'], color = cmap(norm(sat[opt['hmv']])), alpha = opt['alpha'])                
        if opt['hidevio'] == False:
            if novio_flag == False:
                if (opt['hmv'] == 'None') or (opt['hidehmv'] == True):
                    ax.scatter(*zip(*vio[vnames].to_numpy()), marker = opt['mvio'], facecolors = opt['fvio'], label = opt['lvio'], color = opt['cvio'], alpha = opt['alpha'])
                else:
                    ax.scatter(*zip(*vio[vnames].to_numpy()), marker = opt['mvio'], label = opt['lvio'], color = cmap(norm(vio[opt['hmv']])), alpha = opt['alpha'])
        
        # ----- Design space surface/boundary ----- #
        if opt['hidenor'] == False:
            space_size, bF, P, shp = self.envelope(opt['a'])
            if dim == 2:
                for i in range(bF.shape[0]):
                    if i == 0:
                        plt.plot(P[bF][i][:, 0], P[bF][i][:, 1], color = opt['cspace'], label = opt['spacelabel'])
                    else:
                        plt.plot(P[bF][i][:, 0], P[bF][i][:, 1], color = opt['cspace'])

            if dim == 3:
                surf = ax.plot_trisurf(*zip(*P), triangles = bF, color = opt['cspace'], alpha = opt['alphaspace'], label = opt['spacelabel'])
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
                ax.set_xlim([vmin[0] - limfactor*vrange[0], vmax[0] + limfactor*vrange[0]])
                ax.set_ylim([vmin[1] - limfactor*vrange[1], vmax[1] + limfactor*vrange[1]])
                ax.set_zlim([vmin[2] - limfactor*vrange[2], vmax[2] + limfactor*vrange[2]])
        
        
        # Labels
        plt.xlabel(opt['xlabel'])
        plt.ylabel(opt['ylabel'])
        if dim == 3:
            ax.set_zlabel(opt['zlabel'])
            ax.view_init(elev = opt['elev'], azim = opt['azim'])
        if (opt['hmv'] == 'None') or (opt['hidehmv'] == True):
            plt.legend(loc = opt['legloc'])
        
        # Saving
        plt.tight_layout()
        if opt['save_flag']:
            plt.savefig(opt['save_name'], dpi = opt['save_dpi'])
        return self.report
    
    def envelope(self, a = None):
        """
        Create convex hull using MATLAB alphashape
        """
        import matlab.engine
        import numpy as np
        import pandas as pd
        eng = matlab.engine.start_matlab() # Start instance of matlab engine
        ev = eng.eval
        ew = eng.workspace
        
        sat = self.sat
        vio = self.sat
        vnames = self.vnames
        opt = self.opt        
        points = sat[vnames].to_numpy()
        
        ew['points'] = matlab.double(points.tolist()) 
        if a == None:
            a = pd.concat([sat, vio], axis = 0)[vnames].max().product()*opt['amul']
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
    
    def collect_frames(self, anielev = 10, sfolder = 'animation', sname = 'plot', leading_no = 0, sdpi = 100):
        """
        ONLY FOR 3D PLOTS
        Collect .png images for animation of plot by rotating the 3D plot
        720 images will be collected with 0.5 increments for a total of 1 whole rotation
        Args:
            ax ([type]): matplotlib axes
            anielev ([type]): elevation of the figure
            sfolder ([type]): string of the save folder
            sname ([type]): save name of the pictures generated, will be followed by {i:04}.png
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
                plt.savefig(f'{sfolder}{sname}_frame_{leading_no + ii:04}.png', dpi = sdpi)
    
    def flex_space(self, x):
        """
        Plot the flexibility space
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matlab.engine
        eng = matlab.engine.start_matlab() # Start instance of matlab engine
        ev = eng.eval
        ew = eng.workspace
        
        self.all_x.append(x)
        
        ax = self.ax
        shp = self.shp
        opt = self.opt
        sat = self.sat
        vnames = self.vnames
        dim = len(vnames)
        step_change = opt['step_change']
        df = self.df
                        
        # Finding space boundary
        inputs_max = df[vnames].max().to_numpy()
        inputs_min = df[vnames].min().to_numpy()
        inputs_range = inputs_max - inputs_min
        pc = step_change/100
        
        # ----- Plotting ----- #
        ax.scatter(*zip(x), marker = opt['npmarker'], color = opt['npcolor'], label = opt['nplabel'])
        if eng.inShape(shp, matlab.double(list(x))) == False:
            print('x is not inside NOR.')
        else:
            # ----- 2D space ----- #
            if dim == 2:
                flag = False
                while flag == False: # Going outwards
                    fsi   = np.array([[x[0] - pc*inputs_range[0], x[1] - pc*inputs_range[1]],
                                    [x[0] - pc*inputs_range[0], x[1] + pc*inputs_range[1]],
                                    [x[0] + pc*inputs_range[0], x[1] + pc*inputs_range[1]],
                                    [x[0] + pc*inputs_range[0], x[1] - pc*inputs_range[1]]])
                    flag = False in [eng.inShape(shp, matlab.double(list(fsi[i]))) for i in range(fsi.shape[0])]
                    pc += step_change/100

                pc = (step_change/100)/100
                flag = True
                while flag == True: # Going inwards
                    fs = [[fsi[0, 0] + pc*inputs_range[0], fsi[0, 1] + pc*inputs_range[1]],
                        [fsi[1, 0] + pc*inputs_range[0], fsi[1, 1] - pc*inputs_range[1]],
                        [fsi[2, 0] - pc*inputs_range[0], fsi[2, 1] - pc*inputs_range[1]],
                        [fsi[3, 0] - pc*inputs_range[0], fsi[3, 1] + pc*inputs_range[1]]]
                    fs = np.array(fs)
                    flag = False in [eng.inShape(shp, matlab.double(list(fs[i]))) for i in range(fs.shape[0])]
                    pc += (step_change/100)/100
                    
                fs = fs.tolist()
                fs.append(fs[0])
                fs = np.array(fs)
                plt.plot(*zip(*fs), linestyle = opt['fsstyle'], color = opt['fscolor'], label = opt['fslabel'])
                
            # ----- 3D space ----- #
            if dim == 3: 
                flag = False
                while flag == False: # Going outwards
                    fsi = np.array([[x[0] - pc*inputs_range[0], x[1] - pc*inputs_range[1], x[2] - pc*inputs_range[2]],
                                    [x[0] + pc*inputs_range[0], x[1] - pc*inputs_range[1], x[2] - pc*inputs_range[2]],
                                    [x[0] - pc*inputs_range[0], x[1] + pc*inputs_range[1], x[2] - pc*inputs_range[2]],
                                    [x[0] - pc*inputs_range[0], x[1] - pc*inputs_range[1], x[2] + pc*inputs_range[2]],
                                    [x[0] + pc*inputs_range[0], x[1] + pc*inputs_range[1], x[2] - pc*inputs_range[2]],
                                    [x[0] + pc*inputs_range[0], x[1] + pc*inputs_range[1], x[2] + pc*inputs_range[2]],
                                    [x[0] + pc*inputs_range[0], x[1] - pc*inputs_range[1], x[2] + pc*inputs_range[2]],
                                    [x[0] - pc*inputs_range[0], x[1] + pc*inputs_range[1], x[2] + pc*inputs_range[2]]])
                    flag = False in [eng.inShape(shp, matlab.double(list(fsi[i]))) for i in range(fsi.shape[0])]
                    pc += step_change/100

                pc = (step_change/100)/100
                flag = True
                while flag == True: # Going inwards
                    fs = [[fsi[0, 0] + pc*inputs_range[0], fsi[0, 1] + pc*inputs_range[1], fsi[0, 2] + pc*inputs_range[2]],
                        [fsi[1, 0] - pc*inputs_range[0], fsi[1, 1] + pc*inputs_range[1], fsi[1, 2] + pc*inputs_range[2]],
                        [fsi[2, 0] + pc*inputs_range[0], fsi[2, 1] - pc*inputs_range[1], fsi[2, 2] + pc*inputs_range[2]],
                        [fsi[3, 0] + pc*inputs_range[0], fsi[3, 1] + pc*inputs_range[1], fsi[3, 2] - pc*inputs_range[2]],
                        [fsi[4, 0] - pc*inputs_range[0], fsi[4, 1] - pc*inputs_range[1], fsi[4, 2] + pc*inputs_range[2]],
                        [fsi[5, 0] - pc*inputs_range[0], fsi[5, 1] - pc*inputs_range[1], fsi[5, 2] - pc*inputs_range[2]],
                        [fsi[6, 0] - pc*inputs_range[0], fsi[6, 1] + pc*inputs_range[1], fsi[6, 2] - pc*inputs_range[2]],
                        [fsi[7, 0] + pc*inputs_range[0], fsi[7, 1] - pc*inputs_range[1], fsi[7, 2] - pc*inputs_range[2]]]
                    fs = np.array(fs)
                    flag = False in [eng.inShape(shp, matlab.double(list(fs[i]))) for i in range(fs.shape[0])]
                    pc += (step_change/100)/100
                    
                faces = [[fs[0], fs[1], fs[4], fs[2], fs[0]],
                        [fs[0], fs[3], fs[6], fs[1], fs[0]],
                        [fs[3], fs[7], fs[5], fs[6], fs[3]], 
                        [fs[6], fs[1], fs[4], fs[5], fs[6]],
                        [fs[7], fs[3], fs[0], fs[2], fs[7]],
                        [fs[2], fs[4], fs[5], fs[7], fs[2]]]
                for i in faces:
                    plt.plot(*zip(*i[:-1]), color = opt['fscolor'])
                plt.plot(*zip(*faces[-1]), color = opt['fscolor'], label = opt['fslabel'])

            # ----- KPIs ----- #
            rmax = fs.max(axis = 0)
            rmin = fs.min(axis = 0)
            fs_size = (rmax - rmin).prod()
            plusmin = (rmax - rmin)/2
        
            # ----- Heat map variable ----- #
            hmv_fs_R = {'name': opt['hmv'], 'mean': '-', 'max': '-', 'max_sample': '-', 'min': '-',  'min_sample': '-', 'fs_all_samples': '-'}
            no_samples_flag = False
            fs_df = df.copy()
            if opt['hmv'] != 'None':
                fs_df = sat.copy()
                for i in range(dim):
                    fs_df = fs_df[fs_df[vnames[i]] <= rmax[i]]
                for i in range(dim):
                    fs_df = fs_df[fs_df[vnames[i]] >= rmin[i]]
                if fs_df.shape[0] == 0:
                    print('No samples inside flexibility cube available.')
                    no_samples_flag = True
                if no_samples_flag == False:
                    hmv_fs_R['mean']       = fs_df[opt['hmv']].mean()
                    hmv_fs_R['max']        = fs_df[opt['hmv']].max()
                    hmv_fs_R['max_sample'] = fs_df[fs_df[opt['hmv']] == hmv_fs_R['max']]
                    hmv_fs_R['min']        = fs_df[opt['hmv']].min()
                    hmv_fs_R['min_sample'] = fs_df[fs_df[opt['hmv']] == hmv_fs_R['min']]
                    hmv_fs_R['fs_all_samples'] = fs_df
            fs_R = {'rmax': rmax, 'rmin': rmin, 'space_size': fs_size, 'plusmin': plusmin, 'nosam': fs_df.shape[0], 
                    'hmv': hmv_fs_R, 'hmv_sam_flag': no_samples_flag}    
            self.report.update({'fs': fs_R})        
        
        if (opt['hmv'] == 'None') or (opt['hidehmv'] == True):
            plt.legend(loc = opt['legloc'])
        
        return self.report
    
    def send_output(self, output_filename = 'DSI_output.txt', appendix = True):
        """
        Send report of the DSI study as a txt file.
        """
        rp = self.report
        vnames = self.vnames
        f = open(f'{output_filename}', 'w')
        # Headers
        f.write('Design Space Identification\n')
        f.write(f'Dataset name: {output_filename}\n\n')
        f.write(f'No of samples: {self.df.shape[0]}\n')

        f.write('Variables/parameters varied: \n')
        for i, name in enumerate(vnames):
            f.write(f'{name}\n')

        f.write('\nConstraints used: \n')
        for i in self.constraints:
            f.write(f'{i:15} Lower bound: {self.constraints[i][0]:12}     Upper bound: {self.constraints[i][1]:12}\n')


        # DSI results
        f.write(f'\n# ------------------------------ RESULTS ------------------------------ #\n')
        f.write(f'Design space size: {rp["space_size"]:12}\n\n')
        f.write(f'Average {rp["hmv"]["name"]}:    {rp["hmv"]["mean"]:12}\n')
        f.write(f'DS Maximum {rp["hmv"]["name"]}: {rp["hmv"]["max"]:12}\n')
        f.write(f'DS Minimum {rp["hmv"]["name"]}: {rp["hmv"]["min"]:12}\n')
        f.write(f'\n-------------------------------------------------------------------------\n')
        f.write(f'Detailed maximum point: \n')
        if type(rp["hmv"]["max_sample"]) != str:
            f.write(f'{rp["hmv"]["max_sample"].to_string()}\n\n')
        f.write(f'Detailed minimum point: \n')
        if type(rp["hmv"]["min_sample"]) != str:
            f.write(f'{rp["hmv"]["min_sample"].to_string()}')
        f.write(f'\n-------------------------------------------------------------------------\n')

        if len(self.all_x) != 0:
            for i in range(len(self.all_x)):
                x = self.all_x[i]
                rmax = rp['fs']['rmax']
                rmin = rp['fs']['rmin']
                fs_size = rp['fs']['space_size']
                plusmin = rp['fs']['plusmin']

                if x != None:
                    f.write(f'\n\n\n# ------------------------------ Flexibility Space {i+1:03} ------------------------------ #\n')
                    f.write(f'Flexibility space point: \n')
                    for i, vn in enumerate(vnames):
                        f.write(f'{vn}: {x[i]:12}' + '\u00B1' + f'{plusmin[i]:12} Range: {rmax[i] - rmin[i]: 12}\n')
                    f.write(f'Flexibility space size: {fs_size:12}\n')

                    if rp['fs']['hmv_sam_flag']:
                        f.write('No samples inside flexibility cube available.')
                    else:
                        f.write(f'\nNumber of samples inside flexibility cube: {rp["fs"]["nosam"]}\n')
                        f.write(f'Average {rp["fs"]["hmv"]["name"]}:    {rp["fs"]["hmv"]["mean"]:12}\n')
                        f.write(f'DS Maximum {rp["fs"]["hmv"]["name"]}: {rp["fs"]["hmv"]["max"]:12}\n')
                        f.write(f'DS Minimum {rp["fs"]["hmv"]["name"]}: {rp["fs"]["hmv"]["min"]:12}\n')
                        f.write(f'\n-------------------------------------------------------------------------\n')
                        f.write(f'Detailed maximum point: \n')
                        if type(rp["fs"]["hmv"]["max_sample"]) != str:
                            f.write(f'{rp["fs"]["hmv"]["max_sample"].to_string()}\n\n')
                        f.write(f'Detailed minimum point: \n')
                        if type(rp["fs"]["hmv"]["min_sample"]) != str:
                            f.write(f'{rp["fs"]["hmv"]["min_sample"].to_string()}')
                        f.write(f'\nAll samples inside flexibility cube: \n')
                        if type(rp["fs"]["hmv"]["fs_all_samples"]) != str:
                            f.write(f'{rp["fs"]["hmv"]["fs_all_samples"].to_string()}')
                        f.write(f'\n-------------------------------------------------------------------------\n')

        if appendix:
            f.write('\n\n\n\n# ------------------------------ APPENDIX ------------------------------ #\n')
            f.write(f'ALL SATISFIED SAMPLES: \n')
            f.write(f'\n{self.sat.to_string()}\n\n')
            f.write(f'\n-------------------------------------------------------------------------\n')
            f.write(f'ALL VIOLATED SAMPLES: \n')
            f.write(f'\n{self.vio.to_string()}\n\n')

        f.close()