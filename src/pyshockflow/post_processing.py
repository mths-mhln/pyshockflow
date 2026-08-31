import pickle 
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numpy.compat import Path
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from pathlib import Path, WindowsPath
from rich.table import Table
from rich.console import Console

import pyshockflow
from thermoplot.thermoplot import thermoplot_cached
from pyshockflow.driver import Driver
from pyshockflow.config import Config




def make_animations(groupedIterResultsPicklePath: str, maxLength: int, FPS: int, DPI: int) -> None:
    """
    Show the temporal evolution of the solution over the computational comain. 

    Arguments
    ---------
    groupedIterResultsPicklePath : str
        The path to the results pickle file containing the grouped results of each 
        iteration performed during the completed simulation. Intermediate results files
        are merged at the end of the simulation if it converged or reached the maximum 
        simulation time limit. This must be the case for make_animation to make sense. 
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
    with open(groupedIterResultsPicklePath, 'rb') as file:
        groupedIterResults = pickle.load(file)

    # save aliases for the arrays of interest
    xMeshNodes = groupedIterResults["meshData"]['xMeshNodes']
    timeHistory = groupedIterResults["timeHistory"]
    rho = groupedIterResults["fluidState"]['Density']
    u = groupedIterResults["fluidState"]['Velocity']
    p = groupedIterResults["fluidState"]['Pressure']

    # create driver object in order to access the fluid object and compute the internal energy
    config = groupedIterResults["config"]
    with pyshockflow.post_processing.HiddenPrints():
        driver = Driver(config = config)
    e = driver.fluidModel.computeInternalEnergy_p_rho(p, rho)
    _, nTimes = rho.shape
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
        xmin, xmax = plot_limits(xMeshNodes)
        ymin, ymax = plot_limits(field)

        fig, ax = plt.subplots()
        line, = ax.plot([], [], '-C0')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.2)

        def update(iteration):
            line.set_data(xMeshNodes, field[:, iteration])
            ax.set_title(f'Time: {timeHistory[iteration]:.3e} [s]')
            fig.tight_layout()
            return line

        ani = animation.FuncAnimation(fig, update, frames=iterations, blit=False)

        # Save the animation as a video
        ani.save(videoNames[i], writer='ffmpeg', fps=FPS, dpi=DPI)
        print('Video %s Saved' %(videoNames[i]))

    return None



def expansion_device_geometry_plot(config: Config) -> None:
        """
        Plot the expansion device geometry and numerical grid.

        Arguments
        ---------
        config : Config
            Configuration object containing the simulation settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot of the expansion device geometry and numerical grid.
        """
        # instantiate driver object from config file (already most of the necessary functionality)
        # The driver object has internal procedures that extract the device geometry data
        # and mesh upon initialization, and are accessible as attributes 
        # driver.deviceGeometryData and driver.meshData respectively. 
        with pyshockflow.post_processing.HiddenPrints():
            driver = Driver(config = config)

        deviceX = driver.deviceGeometryData ["deviceX"]
        deviceY = driver.deviceGeometryData["deviceY"]

        # Scale plot axes according to the nozzle geometry
        x_scale = np.max(deviceX)
        lengthScale = 2*np.max(deviceY)
        lengthRatio = lengthScale / x_scale

        # plot nozzle
        fig = plt.figure(figsize=(12, 12*lengthRatio))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(deviceX, deviceY, label='Interpolated Nozzle Area', color='blue')
        ax.plot(deviceX, -deviceY, label='Interpolated Nozzle Area', color='blue')
        ax.scatter(deviceX, np.zeros_like(deviceX), color='red', label='Virtual Mesh Nodes', s=0.5)
        ax.set_xlabel('x [m]', fontsize=12)
        ax.set_ylabel('Area [m^2]', fontsize=12)
        ax.set_title('Nozzle Geometry', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        fig.show()
        fig.tight_layout()

        return fig



def unpack_simulation_results(pickleFilePath: type[WindowsPath]) -> dict:
    """
    Extract the expansion path data from the simulation results for a list of specified steps.
    """
    # Load solution data from pickle file
    with open(str(pickleFilePath), 'rb') as file:
        simulationResults = pickle.load(file)

    # instantiate results dictionary for future easy access to simulation results.
    # fluidState and iterIdx are either the final one, or a state and index of an 
    # intermediate iteration before the simulation was terminated due to reaching 
    # either the maximum simulation time limit or converging.
    unpackedSimulationResults = {
        "config": None, 
        "deviceGeometryData": None,
        "meshData": None,
        "(final)fluidState": None,
    }

    # two options: fully finished sim, or partially finished sim. The datastructure of the results
    # files will be slightly different due to the transformation by the groupSingleIterResults 
    # method of Driver (see folder of this file) applies to the results
    unpackedSimulationResults["config"] = simulationResults["config"]
    unpackedSimulationResults["deviceGeometryData"] = simulationResults["deviceGeometryData"]
    unpackedSimulationResults["meshData"] = simulationResults["meshData"]
    if "fluidStateHistory" in simulationResults.keys(): 
        # indicates multi dimensional primitive arrays: indicating merged results file, 
        # processed by the groupSingleIterResults method, indicating sim successfully finished sim
        unpackedSimulationResults["(final)fluidState"] = {}
        for key in simulationResults["fluidStateHistory"].keys():
            unpackedSimulationResults["(final)fluidState"][key] = simulationResults["fluidStateHistory"][key][:,-1]
    else:
        # only partial finished sim. Solution file arrays are 1D
        unpackedSimulationResults["(final)fluidState"] = simulationResults["fluidState"]

    return unpackedSimulationResults




def plot_results(pickleFilePathsList: list[type[WindowsPath]], fluidStateVars: list[str], showNozzleGeometry: bool = False) -> type[plt.Figure]:
    """
    Plot the results of the simulation for a list of specified steps.

    Arguments
    ---------
    pickleFilePathsList : list of str
        The list of pickle files to use for plotting.
    fluidStateVars : list of str
        The list of fluid states to plot. Supported states: 
        "xMeshNodes", "Density", "Pressure", "Velocity", "Mach", "Entropy", 
        "TotalPressure", "Temperature", "TotalTemperature"
    showNozzleGeometry : bool, optional
        Whether to include the nozzle geometry in the plot. Default is False.

    Returns
    -------
    None. Saves the plot in the directory from which the file in which the function 
    is called is executed.
    """
    for fluidStateVar in fluidStateVars:
        # instantiate figure and axes objects
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # instantiate variable to keep track of maximum y value across all steps,
        # to be able to scale the nozzle geometry accordingly.
        max_y = 0

        if type(pickleFilePathsList) is not list:
            raise TypeError("pickleFilePathsList must be a list of pickle file paths." \
            " If you are trying to view the results of a single step, convert the file" \
            " path string to a list, e.g. [pickleFilePath]")
        for pickleFilePath in pickleFilePathsList:
            # extract the nozzle geometry from the last simulation results
            # This can either be the last intermediate step, or the final 
            # grouped results file. get_expansion_data is capable of extracting
            # the data for both. simulationResults is hence either singleIterResult 
            # or groupedIterResults
            unpackedSimulationResults = unpack_simulation_results(pickleFilePath)
            # print(unpackedSimulationResults)

            # instantiate dictionary to store desired fluid states for plotting.
            fluid_state = {}

            # translation dict for automatic axis labeling based on variables user is interested in plotting
            translation_dict = {
                "Density": r'$\rho$ [kg/m³]',
                "Pressure": r'$p$ [Pa]',
                "Velocity": r'$u$ [m/s]',
                "internalEnergy": r'$e$ [J]',
                "Mach": r'$M$',
                "Entropy": r'$s$ [J/kg/K]',
                "Temperature": r'$T$ [K]',
            }

            # depending on the variables of interest, perform the necessary operations
            # simple extractions
            if fluidStateVar == "Density":
                fluid_state[fluidStateVar] = unpackedSimulationResults["(final)fluidState"]['Density'][1:-1]
            elif fluidStateVar == "Pressure":
                fluid_state[fluidStateVar] = unpackedSimulationResults["(final)fluidState"]['Pressure'][1:-1]
            elif fluidStateVar == "Velocity":
                fluid_state[fluidStateVar] = unpackedSimulationResults["(final)fluidState"]['Velocity'][1:-1]
            elif fluidStateVar == "internalEnergy":
                fluid_state[fluidStateVar] = unpackedSimulationResults["(final)fluidState"]['internalEnergy'][1:-1]
            # information that requires additional processing
            elif fluidStateVar == "Mach":
                with pyshockflow.post_processing.HiddenPrints():
                    driver = Driver(config = unpackedSimulationResults["config"])
                fluid_state[fluidStateVar] = driver.fluidModel.computeMach_u_p_rho(
                    unpackedSimulationResults["(final)fluidState"]['Velocity'][1:-1],
                    unpackedSimulationResults["(final)fluidState"]['Pressure'][1:-1],
                    unpackedSimulationResults["(final)fluidState"]['Density'][1:-1]
                )
            elif fluidStateVar == "Entropy":
                with pyshockflow.post_processing.HiddenPrints():
                    driver = Driver(config = unpackedSimulationResults["config"])
                fluid_state[fluidStateVar] = driver.fluidModel.computeEntropy_p_rho(
                    unpackedSimulationResults["(final)fluidState"]['Pressure'][1:-1],
                    unpackedSimulationResults["(final)fluidState"]['Density'][1:-1]
                )
            elif fluidStateVar == "Temperature":
                with pyshockflow.post_processing.HiddenPrints():
                    driver = Driver(config = unpackedSimulationResults["config"])
                fluid_state[fluidStateVar] = driver.fluidModel.computeTemperature_p_rho(
                    unpackedSimulationResults["(final)fluidState"]['Pressure'][1:-1],
                    unpackedSimulationResults["(final)fluidState"]['Density'][1:-1]
                )
            # Extract the y range for the current variable to be plotted, to be able to scale
            # the nozzle geometry accordingly in the plot such that the nozzle geometry I will
            # display in the background is of adequate size.
            if np.abs(fluid_state[fluidStateVar]).max() > max_y:
                max_y = np.abs(fluid_state[fluidStateVar]).max()

            # plot variable of interest and set y label to the variable name using the translation dict
            step = pickleFilePath.stem.split("_")[-1].lstrip('0')
            ax.plot(
                unpackedSimulationResults["meshData"]["xMeshNodes"][1:-1], 
                fluid_state[fluidStateVar], label=r'$iteration=%s$' %(step)
                )
            ax.set_ylabel(translation_dict[fluidStateVar])
            ax.set_xlabel(r"$x$ [m]")

        # plot nozzle scaled to y range
        y_interval = [0, 1.2*max_y]
        # if the user wants to plot the nozzle geometry on the results plot to get an idea
        # of where along the nozzle a certain fluid state occurs...
        if showNozzleGeometry:
            # ... plot the nozzle geometry scaled to y range.
            ax.plot(
                unpackedSimulationResults["meshData"]["xMeshNodes"][1:-1], 
                unpackedSimulationResults["meshData"]["deviceAreaAtMeshNodes"][1:-1]*y_interval[1]*0.3/max(unpackedSimulationResults["meshData"]["deviceAreaAtMeshNodes"][1:-1]), 
                label='Nozzle Geometry', color='gray', alpha=0.5, zorder=-1
                )
        
        # set legend, adjust subplot to make room for legend, save figure
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize = 6)
        fig.subplots_adjust(bottom=0.25)
        out_root = Path("Pictures") 
        out_root.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'Pictures/{fluidStateVar}.pdf', bbox_inches='tight')
        
        # set window title and position on screen
        manager = fig.canvas.manager
        manager.window.wm_geometry("+50+120")
        manager.set_window_title("Nozzle Simulation Results")

    return fig


def thermoplot_expansion_plot(
    thermoplotConfigFilePath: str,
    pickleFilePaths: list[type[WindowsPath]],
    config: type[Config] = None,
    labels: list[str] = None
) -> type[plt.Figure]:
    """
    Plot expansion paths from multiple simulation results on a single thermoplot.

    Args:
        thermoplotConfigFilePath: Path to the thermoplot configuration file.
        pickleFilePaths:          List of paths to pickle files containing simulation results.
        config:                   Optional Config object to extract fluid name from.
        labels:                   Optional list of labels for each expansion path.
                                  Defaults to 'Expansion Path 1', 'Expansion Path 2', etc.
    """
    if labels is None:
        labels = [f"Expansion Path {i + 1}" for i in range(len(pickleFilePaths))]

    if len(labels) != len(pickleFilePaths):
        raise ValueError("Length of 'labels' must match length of 'pickleFilePaths'.")

    # collect entropy and temperature arrays across all pickle files,
    # tracking global min/max to set thermoplot axis limits
    all_entropy = []
    all_temperature = []

    for pickleFilePath in pickleFilePaths:
        unpackedSimulationResults = unpack_simulation_results(pickleFilePath)

        with pyshockflow.post_processing.HiddenPrints():
            driver = Driver(config=unpackedSimulationResults["config"])

        entropy = driver.fluidModel.computeEntropy_p_rho(
            unpackedSimulationResults["(final)fluidState"]['Pressure'][1:-1],
            unpackedSimulationResults["(final)fluidState"]['Density'][1:-1]
        )
        temperature = driver.fluidModel.computeTemperature_p_rho(
            unpackedSimulationResults["(final)fluidState"]['Pressure'][1:-1],
            unpackedSimulationResults["(final)fluidState"]['Density'][1:-1]
        )

        all_entropy.append(entropy)
        all_temperature.append(temperature)

    # adapt thermoplot limits to span all expansion paths with margin
    global_entropy_min = min(s.min() for s in all_entropy)
    global_entropy_max = max(s.max() for s in all_entropy)
    global_temp_min    = min(t.min() for t in all_temperature)
    global_temp_max    = max(t.max() for t in all_temperature)

    thermoplot_overwrite_settings = {
        "S_range": [global_entropy_min * 0.80, global_entropy_max * 1.2],
        "T_range": [global_temp_min    * 0.80, global_temp_max    * 1.2],
    }
    if config is not None:
        thermoplot_overwrite_settings["fluid_name"] = config.fluidName()

    # get plot background
    fig = thermoplot_cached(thermoplotConfigFilePath, thermoplot_overwrite_settings=thermoplot_overwrite_settings)
    ax = fig.get_axes()[0]

    # plot each expansion path with a distinct colour
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (entropy, temperature, label) in enumerate(zip(all_entropy, all_temperature, labels)):
        color = color_cycle[i % len(color_cycle)]
        ax.plot(entropy, temperature, color=color, marker='o', markersize=2, label=label)

    ax.legend()

    return fig



def construct_ideal_expansion_path(pickleFilePath: type[WindowsPath]) -> np.ndarray:
    with open(str(pickleFilePath), 'rb') as file:
        simulationResults = pickle.load(file)
    config = simulationResults['config']

    # if solution is computed using ideal gas model compute also reference from nozzle 
    # theory to check validity of results.
    if config.fluidModelType() == "ideal": 
        xMeshNodes = simulationResults["meshData"]["xMeshNodes"][1:-1]
        deviceAreaRatio =  simulationResults['meshData']['deviceAreaAtMeshNodes'][1:-1] / np.min(simulationResults['meshData']['Area Tube'])
        with pyshockflow.post_processing.HiddenPrints():
            driver = Driver(config=config)
        gammaFluid = driver.fluidModel.gmma
        def machFunction(machLocal, areaRatioLocal, gammaFluid):
            residual = areaRatioLocal - 1/machLocal * (2/(gammaFluid+1) * \
                       (1 + (gammaFluid-1)/2 * machLocal**2))**((gammaFluid+1)/(2*(gammaFluid-1)))
            return residual

        theoreticalMach = np.zeros(len(xMeshNodes))
        idThroat = np.argmin(deviceAreaRatio)
        for iPoint in range(len(xMeshNodes)):
            if iPoint < idThroat:
                theoreticalMach[iPoint] = fsolve(machFunction, 0.1, args=(deviceAreaRatio[iPoint], gammaFluid))[0]
            else:
                theoreticalMach[iPoint] = fsolve(machFunction, 1.2, args=(deviceAreaRatio[iPoint], gammaFluid))[0]
    else:
        raise ValueError("construct_ideal_expansion_path is only applicable for" \
        " ideal gas models. The fluid model type is: %s" %(config.fluidModelType()))
    return np.column_stack((xMeshNodes, theoreticalMach))



def perform_v_and_v(verification_data: dict = None, validation_data: dict = None, simulation_data: dict = None, show_plots: bool = False) -> dict:
    """
    The user provides verification or validation data, with data
    format similar to that resulting from applying the
    unpack_simulation_results function to the simulation results pickle file,
    which can be a singleIterResults or groupedIterResults. The function will
    then compare the simulation results with the verification or validation
    data and return a dictionary containing comparison metrics.

    Arguments
    ---------
    verification_data : dict
        Either a single verification data dict complying to the
        unpack_simulation_results output format, or a dict of such dicts
        where the top-level keys are used as legend labels, e.g.:
            {
                "CaseA.csv": {"meshData": {...}, "(final)fluidState": {...}},
                "CaseB.csv": {"meshData": {...}, "(final)fluidState": {...}},
            }
    validation_data : dict
        Same format as verification_data, but for validation data.
    simulation_data : dict
        Either a single simulation data dict complying to the
        unpack_simulation_results output format, or a dict of such dicts
        where the top-level keys are used as legend labels, e.g.:
            {
                "Simulation A": unpack_simulation_results(...),
                "Simulation B": unpack_simulation_results(...),
            }

    Returns
    -------
    comparison_metrics : dict
        A dictionary containing the comparison metrics. The nesting adapts
        to how many datasets were supplied:
          - 1 V&V dataset,  1 simulation  -> {var: {...}}
          - N V&V datasets, N simulations -> {vnv_label: {sim_label: {var: {...}}}}
        NOTE: Only corresponding pairs are compared (first with first, second with second, etc.)
    """
    _bare_keys = {"meshData", "(final)fluidState", "config"}

    # --- normalise simulation_data to a labelled dict ---
    if any(k in simulation_data for k in _bare_keys):
        simulation_datasets = {"Simulation Data": simulation_data}
    else:
        simulation_datasets = simulation_data

    # --- normalise v_and_v reference data to a labelled dict ---
    if verification_data is not None:
        raw_vnv = verification_data
        vnv_kind = "Verification"
    elif validation_data is not None:
        raw_vnv = validation_data
        vnv_kind = "Validation"
    else:
        raise ValueError("Either verification_data or validation_data must be provided")

    if any(k in raw_vnv for k in _bare_keys):
        v_and_v_datasets = {f"{vnv_kind} Data": raw_vnv}
    else:
        v_and_v_datasets = raw_vnv

    # Check that number of datasets match for paired comparison
    n_vnv = len(v_and_v_datasets)
    n_sim = len(simulation_datasets)
    
    if n_vnv != n_sim:
        raise ValueError(
            f"Number of V&V datasets ({n_vnv}) must equal number of simulation datasets ({n_sim}) "
            f"for paired comparison. If you want all-to-all comparison, use the original function."
        )

    # union of variables across all v_and_v datasets (preserving first-seen order)
    v_and_v_variables = []
    for vnv_data in v_and_v_datasets.values():
        for var in vnv_data["(final)fluidState"].keys():
            if var not in v_and_v_variables:
                v_and_v_variables.append(var)

    # --- interpolate and compute metrics for corresponding pairs only ---
    comparison_metrics = {}

    # Get ordered pairs (first with first, second with second, etc.)
    vnv_items = list(v_and_v_datasets.items())
    sim_items = list(simulation_datasets.items())

    for idx, (vnv_label, vnv_data) in enumerate(vnv_items):
        sim_label, sim_data = sim_items[idx]  # Corresponding simulation
        
        comparison_metrics[vnv_label] = {}
        vnv_vars_here = list(vnv_data["(final)fluidState"].keys())
        sim_metrics = {}
        simulation_interpolated = {}

        same_mesh = np.array_equal(
            vnv_data["meshData"]["xMeshNodes"], sim_data["meshData"]["xMeshNodes"]
        )

        for var in vnv_vars_here:
            if var not in sim_data["(final)fluidState"]:
                continue
            if same_mesh:
                simulation_interpolated[var] = sim_data["(final)fluidState"][var]
            else:
                sim_interpolant = interp1d(
                    sim_data["meshData"]["xMeshNodes"], sim_data["(final)fluidState"][var],
                    kind='linear', fill_value='extrapolate'
                )
                simulation_interpolated[var] = sim_interpolant(vnv_data["meshData"]["xMeshNodes"])

        for var in vnv_vars_here:
            if var in sim_data["(final)fluidState"]:
                abs_error = simulation_interpolated[var] - vnv_data["(final)fluidState"][var]
                relative_error = np.abs(abs_error) / np.abs(vnv_data["(final)fluidState"][var])
                sim_metrics[var] = {
                    'absolute_error': abs_error,
                    'relative_error': relative_error
                }

        if len(sim_metrics) != len(vnv_vars_here):
            missing_keys = set(vnv_vars_here) - set(sim_metrics.keys())
            raise ValueError(
                f"[{vnv_label} vs {sim_label}] Missing keys in simulation data due to "
                f"different naming than V&V data dict keys: {missing_keys}"
            )

        comparison_metrics[vnv_label][sim_label] = sim_metrics

    # --- rich table: one row per (v_and_v dataset, simulation, variable) triple ---
    table = Table(title="Verification and Validation (Paired Comparison)")
    table.add_column("V&V Dataset",        justify="left",  style="yellow",  no_wrap=True)
    table.add_column("Simulation",         justify="left",  style="blue",    no_wrap=True)
    table.add_column("Variable",           justify="left",  style="cyan",    no_wrap=True)
    table.add_column("Max Absolute Error", justify="right", style="magenta")
    table.add_column("Max Relative Error", justify="right", style="green")

    for vnv_label, sim_dict in comparison_metrics.items():
        for sim_label, sim_metrics in sim_dict.items():
            for var, value in sim_metrics.items():
                absolute_error_str = (
                    f"{value['absolute_error']:.6e}" if np.isscalar(value['absolute_error'])
                    else f"{np.max(value['absolute_error']):.6e}"
                )
                relative_error_str = (
                    f"{value['relative_error']:.6e}" if np.isscalar(value['relative_error'])
                    else f"{np.max(value['relative_error']):.6e}"
                )
                table.add_row(vnv_label, sim_label, var, absolute_error_str, relative_error_str)

    console = Console()
    console.print(table)

    # --- plots: ONE figure per variable, all V&V datasets + all simulations overlaid ---
    if show_plots:
        # use the first simulation dataset for nozzle geometry
        first_sim_data = next(iter(simulation_datasets.values()))
        xMeshNodes            = first_sim_data["meshData"]["xMeshNodes"]
        deviceAreaAtMeshNodes = first_sim_data["meshData"]["deviceAreaAtMeshNodes"]

        sim_colors = (
            plt.cm.tab10.colors if len(simulation_datasets) <= 10
            else [plt.cm.tab20(i / len(simulation_datasets)) for i in range(len(simulation_datasets))]
        )
        
        # Use darker colors for V&V data (e.g., Dark2 colormap which is generally darker)
        vnv_colors = (
            plt.cm.Dark2.colors if len(v_and_v_datasets) <= 8
            else [plt.cm.tab20b(i / len(v_and_v_datasets)) for i in range(len(v_and_v_datasets))]
        )

        for var in v_and_v_variables:
            y_candidates = []
            for vnv_data in v_and_v_datasets.values():
                if var in vnv_data["(final)fluidState"]:
                    y_candidates.append(np.max(np.abs(vnv_data["(final)fluidState"][var])))
            for sim_data in simulation_datasets.values():
                if var in sim_data["(final)fluidState"]:
                    y_candidates.append(np.max(np.abs(sim_data["(final)fluidState"][var])))
            max_y = max(y_candidates) if y_candidates else 1.0

            plt.figure(figsize=(10, 5))

            # all v_and_v datasets that have this variable (darker colors)
            for idx, (vnv_label, vnv_data) in enumerate(v_and_v_datasets.items()):
                if var in vnv_data["(final)fluidState"]:
                    # Darken the color by reducing brightness
                    color = vnv_colors[idx % len(vnv_colors)]
                    # Make it even darker for V&V data
                    plt.plot(
                        vnv_data["meshData"]["xMeshNodes"][1:-1],
                        vnv_data["(final)fluidState"][var][1:-1],
                        label=vnv_label, marker='o', linestyle='--', 
                        color=color, linewidth=2.5, markersize=8
                    )

            # all simulation datasets that have this variable (lighter/bright colors)
            for idx, (sim_label, sim_data) in enumerate(simulation_datasets.items()):
                if var in sim_data["(final)fluidState"]:
                    plt.plot(
                        sim_data["meshData"]["xMeshNodes"][1:-1],
                        sim_data["(final)fluidState"][var][1:-1],
                        label=sim_label, marker='x', 
                        color=sim_colors[idx % len(sim_colors)], 
                        linewidth=1.5, markersize=6
                    )

            # nozzle geometry scaled to y range
            plt.plot(
                xMeshNodes,
                deviceAreaAtMeshNodes * 1.2 * max_y * 0.3 / max(deviceAreaAtMeshNodes),
                label='Nozzle Geometry', color='gray', alpha=0.5, zorder=-1
            )
            
            plt.xlabel("xMeshNodes")
            plt.ylabel(var)
            plt.legend()
            plt.grid()
            plt.show()

    # --- unwrap to the simplest matching structure ---
    if n_vnv == 1 and n_sim == 1:
        return next(iter(next(iter(comparison_metrics.values())).values()))
    else:
        # For paired comparison, return the structure as is (vnv_label -> sim_label -> var -> metrics)
        return comparison_metrics

# def perform_v_and_v(verification_data: dict = None, validation_data: dict = None, simulation_data: dict = None, show_plots: bool = False) -> dict:
#     """
#     The user provides verification or validation data, with data
#     format similar to that resulting from applying the 
#     unpack_simulation_results function to the simulation results pickle file, 
#     Which can be a singleIterResults or groupedIterResults. 
#     The function will then compare the simulation results with the verification 
#     or validation data and return a dictionary containing comparison metrics.

#     Arguments
#     ---------
#     verification_data : dict
#         A dictionary containing the verification data, complying to the 
#         unpack_simulation_results function output data format.
#     validation_data : dict
#         A dictionary containing the validation data, complying to the 
#         unpack_simulation_results function output data format.
#     simulation_data : dict
#         A dictionary containing the simulation data, complying to the 
#         unpack_simulation_results function output data format.

#     Returns
#     -------
#     comparison_metrics : dict
#         A dictionary containing the comparison metrics with keys as variable 
#         names and values as dictionaries containing 'error' and 'relative_error'.
#     """
#     # instantiate comparison metrics dict and v_and_v_data dict. 
#     comparison_metrics = {}
#     v_and_v_data = {}

#     # user can specify verification or validation data. 
#     # extract unique fluid statevariables from the available data
#     if verification_data is not None:
#         v_and_v_variables = list(verification_data["(final)fluidState"].keys())
#         v_and_v_data = verification_data
#     elif validation_data is not None:
#         v_and_v_variables = list(validation_data["(final)fluidState"].keys())
#         v_and_v_data = validation_data
#     else:
#         raise ValueError("Either verification_data or validation_data must be provided")

#     # interpolate simulation data to v_and_v_data x-coordinates if they are not already aligned
#     simulation_interpolated = {}
#     if not np.array_equal(v_and_v_data["meshData"]["xMeshNodes"], simulation_data["meshData"]["xMeshNodes"]):
#         for var in v_and_v_variables:
#             if var in simulation_data["(final)fluidState"]:
#                 # Interpolate simulation data to v_and_v_data x-coordinates
#                 sim_interpolant = interp1d(
#                     simulation_data["meshData"]["xMeshNodes"], simulation_data["(final)fluidState"][var], 
#                     kind='linear', fill_value='extrapolate')
#                 simulation_interpolated[var] = sim_interpolant(v_and_v_data["meshData"]["xMeshNodes"])

#     # Extract absolute and relative errors for each variable in v_and_v_data at the shared x-coordinates    
#     for var in v_and_v_variables:
#         if var in simulation_data["(final)fluidState"]:
#             abs_error = simulation_interpolated[var] - v_and_v_data["(final)fluidState"][var]
#             relative_error = np.abs(abs_error) / np.abs(v_and_v_data["(final)fluidState"][var])
#             comparison_metrics[var] = {
#                 'absolute_error': abs_error,
#                 'relative_error': relative_error
#             }
#     if len(comparison_metrics) != len(v_and_v_variables):
#         missing_keys = set(v_and_v_variables) - set(comparison_metrics.keys())
#         raise ValueError(f"Missing keys in simulation data due to different naming than simulation" \
#                           f" data dict keys: {missing_keys}")
    
#     # set up rich table for printing the comparison results to terminal
#     table = Table(title="Verification and Validation")
#     table.add_column("Variable", justify="left", style="cyan", no_wrap=True)
#     table.add_column("Max Absolute Error", justify="right", style="magenta")
#     table.add_column("Max Relative Error", justify="right", style="green")
#     # populate the rich table with the comparison results, displaying only the maximum absolute and relative errors for each variable
#     for key, value in comparison_metrics.items():
#         absolute_error_str = f"{value['absolute_error']:.6e}" if np.isscalar(value['absolute_error']) else f"{np.max(value['absolute_error']):.6e}"
#         relative_error_str = f"{value['relative_error']:.6e}" if np.isscalar(value['relative_error']) else f"{np.max(value['relative_error']):.6e}"
#         table.add_row(key, absolute_error_str, relative_error_str)
#     # display the rich table in the terminal
#     console = Console()
#     console.print(table)
    
#     # Plot the comparison results for each variable
#     if show_plots:
#         # extract nozzle geometry
#         xMeshNodes = simulation_data["meshData"]["xMeshNodes"]
#         deviceAreaAtMeshNodes = simulation_data["meshData"]["deviceAreaAtMeshNodes"]
        
#         # instantiate max_y necessary for rescaling the nozzle geometry to the y range of the variable of interest
#         max_y = 0
        
#         # plot variable progression for each variable in v_and_v_data
#         for var in v_and_v_variables:
#             if var in simulation_data["(final)fluidState"]:
#                 max_y = max(
#                     max_y, np.max(np.abs(simulation_data["(final)fluidState"][var])), 
#                     np.max(np.abs(v_and_v_data["(final)fluidState"][var]))
#                     )                
#                 plt.figure(figsize=(10, 5))
#                 plt.plot(
#                     v_and_v_data["meshData"]["xMeshNodes"][1:-1], v_and_v_data["(final)fluidState"][var][1:-1],
#                     label='Verification/Validation Data', marker='o'
#                     )
#                 plt.plot(
#                     simulation_data["meshData"]["xMeshNodes"][1:-1], simulation_data["(final)fluidState"][var][1:-1], 
#                     label='Simulation Data', marker='x'
#                     )
#                 # plot nozzle scaled to y range
#                 y_interval = [0, 1.2*max_y]
#                 plt.plot(
#                     xMeshNodes, deviceAreaAtMeshNodes*y_interval[1]*0.3/max(deviceAreaAtMeshNodes), 
#                     label='Nozzle Geometry', color='gray', alpha=0.5, zorder=-1
#                     )
#                 plt.title(f'Comparison of {var}')
#                 plt.xlabel("xMeshNodes")
#                 plt.ylabel(var)
#                 plt.legend()
#                 plt.grid()
#                 plt.show()
                
#     return comparison_metrics



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
