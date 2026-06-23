import pickle 
import sys, os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numpy.compat import Path
from scipy.optimize import fsolve
from pathlib import Path, WindowsPath

from thermoplot.thermoplot import thermoplot_cached
from pyshockflow.driver import Driver
from pyshockflow.config import Config






def make_animations(picklePath: str, maxLength: int, FPS: int, DPI: int) -> None:
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
            return line

        ani = animation.FuncAnimation(fig, update, frames=iterations, blit=False)

        # Save the animation as a video
        ani.save(videoNames[i], writer='ffmpeg', fps=FPS, dpi=DPI)
        print('Video %s Saved' %(videoNames[i]))

    return None



def nozzle_geometry_plot(Driver: type[Driver]) -> None:
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
        
        # plot nozzle
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



def results_plots(pickleList: list[type[WindowsPath]], Driver: type[Driver], outputVars: list[str], showNozzleGeometry: bool = False) -> type[plt.Figure]:
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
    # Load nozzle geometry from pickle file, any pickle file will do
    if showNozzleGeometry:
        output_dict = get_expansion_data(pickleList[0])
        nozzleX = output_dict["X Coords"]
        nozzleArea = output_dict["Area Tube"]

    for output_var in outputVars:
        # instantiate figure and axes objects
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # instantiate variable to keep track of maximum y value across all steps, to be able to scale the nozzle geometry accordingly.
        max_y = 0

        if type(pickleList) is not list:
            raise TypeError("pickleList must be a list of pickle file paths. If you are trying to view the results of a single step, convert the file path string to a list, e.g. [pickleFilePath]")
        for pickleFile in pickleList:
            # load expansion data
            output_dict = get_expansion_data(pickleFile)

            # translation dict for automatic axis labeling based on variables user is interested in plotting
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
            # Extract the y range for the current variable to be plotted, to be able to scale the nozzle geometry accordingly in the plot
            # such that the nozzle geometry I will display in the background is of adequate size.
            if np.abs(output_dict[output_var]).max() > max_y:
                max_y = np.abs(output_dict[output_var]).max()

            # plot variable of interest and set y label to the variable name using the translation dict
            step = pickleFile.stem.split("_")[-1].lstrip('0')
            ax.plot(output_dict["X Coords"], output_dict[output_var], label=r'$iteration=%s$' %(step))
            ax.set_ylabel(translation_dict[output_var])

        # plot nozzle scaled to y range
        y_interval = [0, 1.2*max_y]
        if showNozzleGeometry:
            ax.plot(nozzleX, nozzleArea*y_interval[1]*0.3/max(nozzleArea), label='Nozzle Geometry', color='gray', alpha=0.5, zorder=-1)
        
        # set legend, adjust subplot to make room for legend, save figure
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize = 6)
        fig.subplots_adjust(bottom=0.25)
        out_root = Path("Pictures") 
        out_root.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'Pictures/{output_var}.pdf', bbox_inches='tight')
        
        # set window title and position on screen
        manager = fig.canvas.manager
        manager.window.wm_geometry("+50+120")
        manager.set_window_title("Nozzle Simulation Results")

    return fig



def get_expansion_data(pickleFile: type[WindowsPath]) -> dict:
    """
    Extract the expansion path data from the simulation results for a list of specified steps.
    """
    # Load solution data from pickle file
    with open(str(pickleFile), 'rb') as file:
        solution = pickle.load(file)

    # instantiate output dictionary for future easy access to simulation output.
    output_dict = {}

    # two options: fully finished sim, or partially finished sim. The datastructure of the output files will be slightly different due to the 
    # transformation output.py (see folder of this file) applies to the output
    if len(solution['Primitive']['Density'].shape) > 1: # indicates multi dimensional primitive arrays: indicating merged results file, processed by the output object, indicating sim successfully finished
        output_dict["X Coords"] = solution['X Coords'][1:-1]
        output_dict["Area Tube"] = solution['Area Tube'][1:-1]
        output_dict["Density"] = solution['Primitive']["Density"][1:-1,-1]
        output_dict["Pressure"] = solution['Primitive']["Pressure"][1:-1,-1]
        output_dict["Velocity"] = solution['Primitive']["Velocity"][1:-1,-1]
        output_dict["Mach"] = solution['Fluid'].computeMach_u_p_rho(output_dict["Velocity"], output_dict["Pressure"], output_dict["Density"])
        output_dict["Entropy"] = solution['Fluid'].computeEntropy_p_rho(output_dict["Pressure"], output_dict["Density"])
        output_dict["Temperature"] = solution['Fluid'].computeTemperature_p_rho(output_dict["Pressure"], output_dict["Density"])
    else:
        # only partial finished sim. Solution file arrays are 1D
        output_dict["X Coords"] = solution['X Coords'][1:-1]
        output_dict["Area Tube"] = solution['Area Tube'][1:-1]
        output_dict["Density"] = solution['Primitive']["Density"][1:-1]
        output_dict["Pressure"] = solution['Primitive']["Pressure"][1:-1]
        output_dict["Velocity"] = solution['Primitive']["Velocity"][1:-1]
        output_dict["Mach"] = solution['Fluid'].computeMach_u_p_rho(output_dict["Velocity"], output_dict["Pressure"], output_dict["Density"])
        output_dict["Entropy"] = solution['Fluid'].computeEntropy_p_rho(output_dict["Pressure"], output_dict["Density"])
        output_dict["Temperature"] = solution['Fluid'].computeTemperature_p_rho(output_dict["Pressure"], output_dict["Density"])

    return output_dict



def thermoplot_expansion_plot(thermoplot_config_file_path: str, pickleFile: type[WindowsPath], config: type[Config] = None) -> type[plt.Figure]:
    # get expansion data
    output_dict = get_expansion_data(pickleFile)

    # adapt thermoplot limits to center around the expansion path
    thermoplot_overwrite_settings = {}
    thermoplot_overwrite_settings["S_range"] = [output_dict["Entropy"].min()*0.80, output_dict["Entropy"].max()*1.2]
    thermoplot_overwrite_settings["T_range"] = [output_dict["Temperature"].min()*0.80, output_dict["Temperature"].max()*1.2]
    # get fluid name from config file
    if config is not None:
        thermoplot_overwrite_settings["fluid_name"] = config.getFluidName()

    # get plot background
    fig = thermoplot_cached(thermoplot_config_file_path, thermoplot_overwrite_settings=thermoplot_overwrite_settings)

    # get expansion data
    output_dict = get_expansion_data(pickleFile)

    # get plot axes from fig and plot expansion path on top of thermoplot
    ax = fig.get_axes()[0]
    ax.plot(output_dict["Entropy"], output_dict["Temperature"], color='red', marker='o', markersize=2, label='Expansion Path')

    return fig



def construct_ideal_expansion_path(pickleFile: type[WindowsPath]) -> np.ndarray:
    with open(str(pickleFile), 'rb') as file:
        solution = pickle.load(file)
    config = solution['Configuration']
    
    # if solution is computed using ideal gas model compute also reference from nozzle theory to check validity of results.
    if config.getFluidModel() == "ideal" and i==0: 
        xArea = solution['X Coords']
        areaRatio =  solution['Area Tube'] / np.min(solution['Area Tube'])
        gammaFluid = solution['Fluid'].gmma
        def machFunction(machLocal, areaRatioLocal, gammaFluid):
            residual = areaRatioLocal - 1/machLocal * (2/(gammaFluid+1) * (1 + (gammaFluid-1)/2 * machLocal**2))**((gammaFluid+1)/(2*(gammaFluid-1)))
            return residual

        machTheory = np.zeros(len(xArea))
        idThroat = np.argmin(areaRatio)
        for iPoint in range(len(xArea)):
            if iPoint < idThroat:
                machTheory[iPoint] = fsolve(machFunction, 0.1, args=(areaRatio[iPoint], gammaFluid))[0]
            else:
                machTheory[iPoint] = fsolve(machFunction, 1.2, args=(areaRatio[iPoint], gammaFluid))[0]

        return np.column_stack((xArea, machTheory))



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
