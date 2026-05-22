import pickle 
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.compat import Path



def make_animations(picklePath, maxLength, FPS, DPI):
    """
    Show the temporal evolution of the solution over the computational comain. 

    Arguments
    ---------
    picklePath : str
        The path to the pickle file of the simulation. Contains dictionary of 2D np arrays for each primitive variable, (space, time)
    maxLength : int
        The maximum number of time steps to include in the animation.
    FPS : int
        The frames per second for the animation.
    DPI : int
        The dots per inch for the saved video.

    Returns
    -------
    None. Saves the videos in the directory from which the file is executed.
    """
    # open the pickle file
    with open(picklePath, 'rb') as file:
        solution = pickle.load(file)

    # save aliases for the arrays of interest
    x = solution['X Coords']
    time = solution['Time']
    rho = solution['Primitive']['Density']
    u = solution['Primitive']['Velocity']
    p = solution['Primitive']['Pressure']
    e = solution['Fluid'].computeStaticEnergy_p_rho(p, rho)
    nPoints, nTimes = rho.shape
    iterations = np.linspace(0, nTimes-1, num=maxLength, dtype=int)

    fields = [rho, u, p, e]
    labels = ['Density [kg/m3]', 'Velocity [m/s]', 'Pressure [Pa]', 'Energy [J]' ]
    videoNames = ['Density.mp4', 'Velocity.mp4', 'Pressure.mp4', 'Energy.mp4']

    # PLOTS AND VIDEO
    def plot_limits(f, extension=0.05):
        max = f.max()
        min = f.min()
        left = min-(max-min)*extension
        right = max+(max-min)*extension
        return left, right


    for i,field in enumerate(fields):
        xmin, xmax = plot_limits(x)
        ymin, ymax = plot_limits(field)

        fig, ax = plt.subplots()
        line, = ax.plot([], [], '-C0')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.2)

        def update(iteration):
            line.set_data(x, field[:, iteration])
            ax.set_title(f'Time: {time[iteration]:.3e} [s]')
            fig.tight_layout()
            return line,

        ani = animation.FuncAnimation(fig, update, frames=iterations, blit=False)

        # Save the animation as a video
        ani.save(videoNames[i], writer='ffmpeg', fps=FPS, dpi=DPI)
        print('Video %s Saved' %(videoNames[i]))

    return None



def plotNozzleGeometry(Driver):
        """
        Plot the nozzle geometry and numerical grid.

        Arguments
        ---------
        Driver : Driver
            The Driver object capable of extracting the nozzle geometry and constructing the 1D mesh

        Returns
        -------
        None. Saves the plot in the directory from which the file is executed.
        """
        nozzleData = np.loadtxt(Driver.config.getNozzleFilePath(), skiprows=1, delimiter=',', dtype=float)
        nozzleX = nozzleData[:,0]
        nozzleArea = nozzleData[:,1]

        # Linear interpolation with external filling set to area Reference (=Tube area)
        interpolatedNozzleArea = np.interp(Driver.xNodesVirtual, nozzleX, nozzleArea, left=nozzleData[0,1], right=nozzleData[-1,1])

        # Scale plot axes according to the nozzle geometry
        x_scale = Driver.xNodesVirtual[-1]
        area_scale = 2*max(interpolatedNozzleArea)
        ratio = area_scale / x_scale
        
        fig = plt.figure(figsize=(12, 12*ratio))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(Driver.xNodesVirtual, interpolatedNozzleArea, label='Interpolated Nozzle Area', color='blue')
        ax.plot(Driver.xNodesVirtual, -interpolatedNozzleArea, label='Interpolated Nozzle Area', color='blue')
        ax.scatter(Driver.xNodesVirtual, np.zeros_like(Driver.xNodesVirtual), color='red', label='Virtual Mesh Nodes', s=0.5)
        ax.set_xlabel('x [m]', fontsize=12)
        ax.set_ylabel('Area [m^2]', fontsize=12)
        ax.set_title('Nozzle Geometry', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        fig.show()
        fig.tight_layout()
        # fig.savefig("nozzle_geometry.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.show()
        return None



def plotResults(pickleList, Driver, outputVars, showNozzleGeometry=False):
    """
    Plot the results of the simulation for a list of specified steps.

    Arguments
    ---------
    pickleList : list of str
        The list of pickle files to use for plotting.
    Driver : Driver
        The Driver object capable of extracting the nozzle geometry and constructing the 1D mesh
    outputVars : list of str
        The list of output variables to plot. Supported variables: "X Coords", "Density", "Pressure", "Velocity", "Mach", "Entropy", "TotalPressure", "Temperature", "TotalTemperature"
    showNozzleGeometry : bool, optional
        Whether to include the nozzle geometry in the plot. Default is False.

    Returns
    -------
    None. Saves the plot in the directory from which the file in which the function is called is executed.
    """
    ### Load nozzle geometry
    if showNozzleGeometry:
        nozzleData = np.loadtxt(Driver.config.getNozzleFilePath(), skiprows=1, delimiter=',', dtype=float)
        nozzleX = nozzleData[:,0]
        nozzleArea = nozzleData[:,1]
        interpolatedNozzleArea = np.interp(Driver.xNodesVirtual, nozzleX, nozzleArea, left=nozzleData[0,1], right=nozzleData[-1,1])

    for output_var in outputVars:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, pickleFile in enumerate(pickleList):
            ### Load solution data from pickle file
            with open(pickleFile, 'rb') as file:
                solution = pickle.load(file)
            output_dict = {}
            try:
                solution['Primitive']['Density'].shape[1] # indicates multi dimensional primitive arrays: indicating merged results file, processed by the output object, indicating sim successfully finished
                output_dict["X Coords"] = solution['X Coords'][1:-1]
                output_dict["Density"] = solution['Primitive']["Density"][1:-1,-1]
                output_dict["Pressure"] = solution['Primitive']["Pressure"][1:-1,-1]
                output_dict["Velocity"] = solution['Primitive']["Velocity"][1:-1,-1]
                output_dict["Mach"] = solution['Fluid'].computeMach_u_p_rho(output_dict["Velocity"], output_dict["Pressure"], output_dict["Density"])
                output_dict["Entropy"] = solution['Fluid'].computeEntropy_p_rho(output_dict["Pressure"], output_dict["Density"])
                output_dict["TotalPressure"] = solution['Fluid'].computeTotalPressure_p_M(output_dict["Pressure"], output_dict["Mach"])
                output_dict["Temperature"] = solution['Fluid'].computeTemperature_p_rho(output_dict["Pressure"], output_dict["Density"])
                output_dict["TotalTemperature"] = solution['Fluid'].computeTotalTemperature_T_M(output_dict["Temperature"], output_dict["Mach"])
            except:
                # only partial finished sim. Solution file arrays are 1D
                output_dict["X Coords"] = solution['X Coords'][1:-1]
                output_dict["Density"] = solution['Primitive']["Density"][1:-1]
                output_dict["Pressure"] = solution['Primitive']["Pressure"][1:-1]
                output_dict["Velocity"] = solution['Primitive']["Velocity"][1:-1]
                output_dict["Mach"] = solution['Fluid'].computeMach_u_p_rho(output_dict["Velocity"], output_dict["Pressure"], output_dict["Density"])
                output_dict["Entropy"] = solution['Fluid'].computeEntropy_p_rho(output_dict["Pressure"], output_dict["Density"])
                output_dict["Temperature"] = solution['Fluid'].computeTemperature_p_rho(output_dict["Pressure"], output_dict["Density"])

            ### Load figure
            translation_dict = {
                "X Coords": r"$x$ [m]",
                "Density": r'$\rho$ [kg/m³]',
                "Pressure": r'$p$ [Pa]',
                "Velocity": r'$u$ [m/s]',
                "Mach": r'$M$',
                "Entropy": r'$s$ [J/kg/K]',
                "TotalPressure": r'$p_0$ [Pa]',
                "Temperature": r'$T$ [K]',
                "TotalTemperature": r'$T_0$ [K]'
            }
            y_interval = [0, 1.2*max(output_dict[output_var])]
            ax.plot(output_dict["X Coords"], output_dict[output_var], label=r'$iteration=%s$' %(re.search(r".+?step_(\d+).pik", pickleFile).group(1)))
            ax.set_ylabel(translation_dict[output_var])

        # plot nozzle scaled to y range
        if showNozzleGeometry:
            ax.plot(Driver.xNodesVirtual, interpolatedNozzleArea*y_interval[1]*0.3/max(interpolatedNozzleArea), label='Nozzle Geometry', color='gray', alpha=0.5, zorder=-1)
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize = 6)
        fig.subplots_adjust(bottom=0.25)
        out_root = Path("Pictures") 
        out_root.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'Pictures/{output_var}.pdf', bbox_inches='tight')
        
        manager = fig.canvas.manager
        manager.window.wm_geometry("+50+120")
        manager.set_window_title("Nozzle Simulation Results")

    plt.show()   
            
            
    