import pickle 
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.compat import Path
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from pathlib import Path, WindowsPath
from rich.table import Table
from rich.console import Console
from dataclasses import dataclass
from typing import Optional, Sequence, Union
PathLike = Union[str, Path]

import pyshockflow
import pyshockflow.post_processing
from thermoplot.thermoplot import thermoplot_cached
from thermoplot.configthermoplot import ConfigThermoplot
from pyshockflow.driver import Driver
from pyshockflow.config import Config


# ================================
#  Helper Functions
# ================================
def collect_results_by_folder(resultsFolders: list[str]) -> dict:
    """
    Convert result folder paths into a nested dictionary, where each subdictionary contains the 
    simulation results for the corresponding folder. 
    
    Returns
    -------
    dict
        A dictionary containing the collected simulation results.
    """
    # convert resultsFolders to windowsPath objects
    resultsFolders = [Path(folder) for folder in resultsFolders]

    #instantiate dictionary to store the results histories for each folder. 
    simulationResultsHistories = {}

    for resultsFolder in resultsFolders:
        # check if results files have already been grouped
        if Path(resultsFolder / "Results.pik").is_file():
            with open(str(Path(resultsFolder / "Results.pik")), 'rb') as file:
                simulationResultsHisotry = pickle.load(file)
        else:
            # if not, apply driver assembleResultHistory() method to the unfinished folder
            # to create a dictionary of similar structure to the grouped results file.
            # this will simplify future operations. 
            simulationResultsHisotry = Driver.assembleResultHistory(resultsFolder, verbose=False)
        simulationResultsHistories[resultsFolder] = simulationResultsHisotry

    return simulationResultsHistories



def extract_results_at_iteration_indexes(simulationResultsHistories: dict, iterationIndexes: list[int]) -> dict:
    """
    Extract the simulation results at specified iteration indexes from the collected results histories.

    Parameters
    ----------
    simulationResultsHistories : dict
        A dictionary containing the collected simulation results for each folder.
    iterationIndexes : list of int
        A list of iteration indexes to extract from each results history.

    Returns
    -------
    dict
        A dictionary containing the extracted simulation results at the specified iteration indexes.
    """
    # extact the iterIdx of interest from each of the results histories. and convert the 
    # collectdResultsHistories dict from a nested dict with subdicts according to the dicts
    # resulting form the assembleResultHistory() method to those resulting from the 
    # saveSingleIterResult() method. Instantiate a new dict with clearer name for this
    simulationResultsAtIterIdxs = {}
    
    for iterIdx, (resultsFolder, iterResultsHistory) in zip(iterationIndexes, simulationResultsHistories.items()):
        # Check if information is available for the specified iteration index
        if iterIdx not in iterResultsHistory["iterIdxHistory"]:
            raise ValueError(f"{resultsFolder}: No results available for iteration index {iterIdx}. Available iteration indexes are: {iterResultsHistory['iterIdxHistory']}")

        # check if iterationIdxs are specified for all result folders. If not, raise an error.
        if len(iterationIndexes) != len(simulationResultsHistories):
            raise ValueError(f"Number of specified iteration indexes ({len(iterationIndexes)}) does not match the number of result folders ({len(simulationResultsHistories)}). Please specify an iteration index for each result folder.")
        
        # Extract the fluid state of interest for the specified iteration index
        # together with time and iteration index.
        j = np.where(iterResultsHistory["iterIdxHistory"] == iterIdx)[0][0]
        fluidStateAtIterIdx = {key: iterResultsHistory["fluidStateHistory"][key][:, j] for key in iterResultsHistory["fluidStateHistory"].keys()}
        timeAtIterIdx = iterResultsHistory["timeHistory"][j]
        iterIdxAtIterIdx = iterResultsHistory["iterIdxHistory"][j]

        # Fill new dictionary with the extracted information, to be used for plotting.
        simulationResultsAtIterIdxs[resultsFolder] = {
            "config": iterResultsHistory["config"],
            "deviceGeometryData": iterResultsHistory["deviceGeometryData"],
            "meshData": iterResultsHistory["meshData"],
            "iterIdx": iterIdxAtIterIdx,
            "time": timeAtIterIdx,
            "fluidState": fluidStateAtIterIdx
        }

    return simulationResultsAtIterIdxs





# ========================================================
#  Simple Plot showing the nozzle geometry and meshnodes
# ========================================================
def expansion_device_geometry_plot(config: Config = None, resultsFolder: dict = None) -> None:
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
        if config is not None:
            # instantiate driver object from config file (already most of the necessary functionality)
            # The driver object has internal procedures that extract the device geometry data
            # and mesh upon initialization, and are accessible as attributes 
            # driver.deviceGeometryData and driver.meshData respectively. 
            with pyshockflow.post_processing.HiddenPrints():
                driver = Driver(config = config)
            deviceX = driver.deviceGeometryData["deviceX"]
            deviceY = driver.deviceGeometryData["deviceY"]
            meshX = driver.meshData["xMeshNodes"]

        elif resultsFolder is not None:
            simulationResultsHistory = collect_results_by_folder([resultsFolder])
            deviceX = simulationResultsHistory[resultsFolder]["deviceGeometryData"]["deviceX"]
            deviceY = simulationResultsHistory[resultsFolder]["deviceGeometryData"]["deviceY"]
            meshX = simulationResultsHistory[resultsFolder]["meshData"]["xMeshNodes"]

        # Scale plot axes according to the nozzle geometry
        x_scale = np.max(deviceX)
        lengthScale = 2*np.max(deviceY)
        lengthRatio = lengthScale / x_scale

        # plot nozzle
        fig = plt.figure(figsize=(12, 12*lengthRatio))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(deviceX, deviceY, label='Interpolated Nozzle Area', color='blue')
        ax.plot(deviceX, -deviceY, label='Interpolated Nozzle Area', color='blue')
        ax.scatter(meshX, np.zeros_like(meshX), color='red', label='Virtual Mesh Nodes', s=0.5)
        ax.set_xlabel('x [m]', fontsize=12)
        ax.set_ylabel('Area [m^2]', fontsize=12)
        ax.set_title('Nozzle Geometry', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        fig.show()
        fig.tight_layout()

        return fig



# def generate_fluid_state_var_profile_figs(
#     simulationResultsHistories: dict[str, dict], fluidStateVarNames: list[str],
#     iterationIndexes: list[int],
#     fluidStateVarsVerificationDataPaths: list[tuple[str]] = None, 
#     fluidStateVarsValidationDataPaths: list[tuple[str]] = None,
#     simulationLegendLabels: list[str]=None, verificationLegendLabels: list[str]=None, 
#     validationLegendLabels: list[str]=None, showNozzleGeometry: bool = False
#     ) -> type[plt.figure]:
#     """
    
#     """
#     # extact the iterIdx of interest from each of the results histories. and convert the 
#     # collectdResultsHistories dict from a nested dict with subdicts according to the dicts
#     # resulting form the assembleResultHistory() method to those resulting from the 
#     # saveSingleIterResult() method. Instantiate a new dict with clearer name for this
#     simulationResultsAtIterIdxs = extract_results_at_iteration_indexes(simulationResultsHistories, iterationIndexes)

#     # translation dict for automatic axis labeling based on variables user is interested in plotting
#     translation_dict = {
#         "Density": r'$\rho$ [kg/m³]',
#         "Pressure": r'$p$ [Pa]',
#         "Velocity": r'$u$ [m/s]',
#         "internalEnergy": r'$e$ [J]',
#         "Mach": r'$M$',
#         "Entropy": r'$s$ [J/kg/K]',
#         "Temperature": r'$T$ [K]',
#     }

#     # instantiate list of fluidStateVarProfileFigs to store the figures for each variable of interest, to be returned to the user.
#     fluidStateVarProfileFigs = []

#     # check if iterationIndexes is specified for all result folders. If not, raise an error.
#     # the same condition holds for simulationLegendLabels
#     if len(iterationIndexes) != len(simulationResultsHistories):
#         raise ValueError("iterationIndexes of interest must be specified for all result folders.")
#     if len(simulationLegendLabels) != len(simulationResultsHistories):
#         raise ValueError("simulationLegendLabels must be specified for all result folders.")

#     # Parse fluidStateVarVerificationDataPaths and fluidStateVarValidationDataPaths input
#     # to ensure they 
#     # 1) exist, and
#     # 1) contain the same number of elements as fluidStateVarNames
#     if fluidStateVarsVerificationDataPaths is None:
#         fluidStateVarsVerificationDataPaths = tuple(None for _ in fluidStateVarNames)
#     if fluidStateVarsValidationDataPaths is None:
#         fluidStateVarsValidationDataPaths = tuple(None for _ in fluidStateVarNames)

#     if fluidStateVarsVerificationDataPaths is not None:
#         if len(fluidStateVarsVerificationDataPaths) != len(fluidStateVarNames):
#             raise ValueError(
#                 "fluidStateVarsVerificationDataPaths must have one entry (a tuple of paths, "
#                 "possibly empty) per entry in fluidStateVarNames."
#             )
#     if fluidStateVarsValidationDataPaths is not None:
#         if len(fluidStateVarsValidationDataPaths) != len(fluidStateVarNames):
#             raise ValueError(
#                 "fluidStateVarsValidationDataPaths must have one entry (a tuple of paths, "
#                 "possibly empty) per entry in fluidStateVarNames."
#             )

#     # loop over 
#     for fluidStateVarName, fluidStateVarVerificationDataPaths, fluidStateVarValidationDataPaths in \
#     zip(fluidStateVarNames, fluidStateVarsVerificationDataPaths, fluidStateVarsValidationDataPaths):
#         # instantiate figure and axes objects
#         fig, ax = plt.subplots(figsize=(12, 6))

#         # instantiate variable to keep track of maximum y value across all steps,
#         # to be able to scale the nozzle geometry accordingly.
#         max_y = []

#         # extract verification and validation data corresponding to this 
#         # variable of interest, if provided by the user.
#         if fluidStateVarVerificationDataPaths is not None:
#             fluidStateVarVerificationData = _process_v_and_v_data(fluidStateVarVerificationDataPaths)
#         else:
#             fluidStateVarVerificationData = None
#         if fluidStateVarValidationDataPaths is not None:
#             fluidStateVarValidationData = _process_v_and_v_data(fluidStateVarValidationDataPaths)
#         else:
#             fluidStateVarValidationData = None

#         # check that each of the v_and_v data dictionaries carry v and v data on
#         # the same variable of interest as the one currently being processed in this loop.
#         if fluidStateVarVerificationDataPaths is not None:
#             if not all(fluidStateVarName in list(data.keys())[1] for data in fluidStateVarVerificationData):
#                 raise ValueError(f"""
#                 Verification data provided for variable {fluidStateVarName}, but not all verification data files contain this variable. 
#                 Verification data contains data on variables: {[list(data.keys())[1] for data in fluidStateVarVerificationData]}. 
#                 Please check the verification data files and ensure that they contain data on the variable of interest: {fluidStateVarName}.
#                 """)

#         if fluidStateVarValidationDataPaths is not None:
#             if not all(fluidStateVarName in list(data.keys())[1] for data in fluidStateVarValidationData):
#                 raise ValueError(f"""
#                 Validation data provided for variable {fluidStateVarName}, but not all validation data files contain this variable. 
#                 Validation data contains data on variables: {[list(data.keys())[1] for data in fluidStateVarValidationData]}. 
#                 Please check the validation data files and ensure that they contain data on the variable of interest: {fluidStateVarName}.
#                 """)

#         # if multiple results folders are specified, the results of each will be
#         # plotted on the same figure for comparison. This is only possible if the 
#         # original simulations for which results are stored in the results folders
#         # were performed with the same nozzle geometry. 
#         nozzleGeometries = []
#         for resultsFolder, resultsAtIterIdx in simulationResultsAtIterIdxs.items():
#             nozzleGeometries.append(resultsAtIterIdx["deviceGeometryData"]["deviceY"])
#         if not all(np.array_equal(nozzleGeometries[0], nozzleGeometry) for nozzleGeometry in nozzleGeometries):
#             raise ValueError("""
#             Multiple results folders were specified. The generate_fluid_state_var_profile_figs()
#             function is written to plot the results of multiple simulations on the same figure for comparison.
#             For proper functionality, the nozzle geometries of the simulations must be identical. 
#             For simulations with different nozzle geometries, please call the generate_fluid_state_var_profile_figs() 
#             function separately for each results folder.
#             """)

#         for i, (resultsFolder, resultsAtIterIdx) in enumerate(simulationResultsAtIterIdxs.items()):

#             # depending on the variables of interest, perform the necessary operations
#             # simple extractions
#             if fluidStateVarName == "Density":
#                 fluidStateVarProfile = resultsAtIterIdx["fluidState"]['Density'][1:-1]
#             elif fluidStateVarName == "Pressure":
#                 fluidStateVarProfile = resultsAtIterIdx["fluidState"]['Pressure'][1:-1]
#             elif fluidStateVarName == "Velocity":
#                 fluidStateVarProfile = resultsAtIterIdx["fluidState"]['Velocity'][1:-1]
#             elif fluidStateVarName == "internalEnergy":
#                 fluidStateVarProfile = resultsAtIterIdx["fluidState"]['internalEnergy'][1:-1]
#             # information that requires additional processing
#             elif fluidStateVarName == "Mach":
#                 with pyshockflow.post_processing.HiddenPrints():
#                     driver = Driver(config = resultsAtIterIdx["config"])
#                 fluidStateVarProfile = driver.fluidModel.computeMach_u_p_rho(
#                     resultsAtIterIdx["fluidState"]['Velocity'][1:-1],
#                     resultsAtIterIdx["fluidState"]['Pressure'][1:-1],
#                     resultsAtIterIdx["fluidState"]['Density'][1:-1]
#                 )
#             elif fluidStateVarName == "Entropy":
#                 with pyshockflow.post_processing.HiddenPrints():
#                     driver = Driver(config = resultsAtIterIdx["config"])
#                 fluidStateVarProfile = driver.fluidModel.computeEntropy_p_rho(
#                     resultsAtIterIdx["fluidState"]['Pressure'][1:-1],
#                     resultsAtIterIdx["fluidState"]['Density'][1:-1]
#                 )
#             elif fluidStateVarName == "Temperature":
#                 with pyshockflow.post_processing.HiddenPrints():
#                     driver = Driver(config = resultsAtIterIdx["config"])
#                 fluidStateVarProfile = driver.fluidModel.computeTemperature_p_rho(
#                     resultsAtIterIdx["fluidState"]['Pressure'][1:-1],
#                     resultsAtIterIdx["fluidState"]['Density'][1:-1]
#                 )

#             # Extract the y range for the current variable to be plotted, to be able to scale
#             # the nozzle geometry accordingly in the plot such that the nozzle geometry I will
#             # display in the background is of adequate size.
#             max_y.append(np.abs(fluidStateVarProfile).max())

#             # plot variable of interest and set y label to the variable name using the translation dict
#             step = resultsAtIterIdx["iterIdx"]
#             ax.plot(
#                 resultsAtIterIdx["meshData"]["xMeshNodes"][1:-1], 
#                 fluidStateVarProfile, label=r'$%s: iteration=%s$' %(simulationLegendLabels[i], step)
#                 )
#             ax.set_ylabel(translation_dict[fluidStateVarName])
#             ax.set_xlabel(r"$x$ [m]")

#             # set legend, adjust subplot to make room for legend, save figure
#             fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize = 6)
#             fig.subplots_adjust(bottom=0.25)
#             out_root = Path("Pictures") 
#             out_root.mkdir(parents=True, exist_ok=True)
#             plt.savefig(f'Pictures/{fluidStateVarName}.pdf', bbox_inches='tight')
            
#             # set window title and position on screen
#             manager = fig.canvas.manager
#             manager.window.wm_geometry("+50+120")
#             manager.set_window_title(f"{resultsFolder}: Simulation Results")

#         # plot nozzle scaled to y range
#         max_y = max(max_y)
#         y_interval = [0, 1.2*max_y]
#         # if the user wants to plot the nozzle geometry on the results plot to get an idea
#         # of where along the nozzle a certain fluid state occurs...
#         if showNozzleGeometry:
#             # ... plot the nozzle geometry scaled to y range, with legend shifted down to avoid overlap with the nozzle geometry plot
#             ax.plot(
#                 resultsAtIterIdx["meshData"]["xMeshNodes"][1:-1], 
#                 resultsAtIterIdx["meshData"]["deviceAreaAtMeshNodes"][1:-1]*y_interval[1]*0.3/max(resultsAtIterIdx["meshData"]["deviceAreaAtMeshNodes"][1:-1]), 
#                 label='Nozzle Geometry', color='gray', alpha=0.5, zorder=-1
#                 )

#         fluidStateVarProfileFigs.append(fig)

#         # plot v and v data if provided by the user
#         if fluidStateVarVerificationData is not None:
#             for i, verificationData in enumerate(fluidStateVarVerificationData):
#                 ax.scatter(
#                     verificationData["deviceX"], 
#                     verificationData[list(verificationData.keys())[1]], 
#                     label=verificationLegendLabels[i], marker='x', color='black'
#                     )
#         if fluidStateVarValidationData is not None:
#             for i, validationData in enumerate(fluidStateVarValidationData):
#                 ax.scatter(
#                     validationData["deviceX"], 
#                     validationData[list(validationData.keys())[1]], 
#                     label=validationLegendLabels[i], marker='o', color='black'
#                     )
            
#     return fluidStateVarProfileFigs







# ========================================
#  Fluid state variable profile plots
# ========================================

# ----------------------------
#  Helpers 
# ----------------------------
@dataclass
class VandVSpec:
    """A set of verification or validation data files, keyed by fluid state
    variable name. Variables with no data simply have no key.

    Attributes
    ----------
    paths:
        Mapping from fluid state variable name (must match an entry in
        `fluidStateVarNames`) to a single path, or a sequence of paths, for
        that variable's data file(s).
    legend_labels:
        Optional mapping from fluid state variable name to a matching label,
        or sequence of labels, one per path given for that variable in
        `paths`. If omitted entirely, or omitted for a given variable, each
        path's file stem is used as its legend label instead.

    Examples
    --------
    >>> VandVSpec(paths={"Pressure": "data/p_ref.csv"})
    >>> VandVSpec(
    ...     paths={"Pressure": ("data/p_ref.csv", "data/p_ref2.csv")},
    ...     legend_labels={"Pressure": ("Analytical", "Experimental")},
    ... )
    """

    paths: dict[str, Union[PathLike, Sequence[PathLike]]]
    legend_labels: Optional[dict[str, Union[str, Sequence[str]]]] = None

    def __post_init__(self) -> None:
        # Normalize every entry in `paths` to a tuple of paths.
        normalized_paths: dict[str, tuple] = {}
        for var_name, entry in self.paths.items():
            if isinstance(entry, (str, Path)):
                normalized_paths[var_name] = (entry,)
            else:
                normalized_paths[var_name] = tuple(entry)
        self.paths = normalized_paths

        # Normalize / default legend labels for every variable in `paths`.
        raw_labels = self.legend_labels or {}
        normalized_labels: dict[str, tuple] = {}
        for var_name, var_paths in self.paths.items():
            labels = raw_labels.get(var_name)
            if labels is None:
                labels = tuple(Path(p).stem for p in var_paths)
            elif isinstance(labels, str):
                labels = (labels,)
            else:
                labels = tuple(labels)

            if len(labels) != len(var_paths):
                raise ValueError(
                    f"VandVSpec: variable '{var_name}' has {len(var_paths)} "
                    f"data path(s) but {len(labels)} legend label(s); these "
                    f"must match one-to-one."
                )
            normalized_labels[var_name] = labels
        self.legend_labels = normalized_labels

    def for_variable(self, var_name: str) -> tuple[tuple, tuple]:
        """Return `(paths, legend_labels)` for `var_name`, or `((), ())` if
        this spec has no data for that variable."""
        return self.paths.get(var_name, ()), self.legend_labels.get(var_name, ())


def _process_v_and_v_data(v_and_v_data_paths: list[str]) -> list[dict, dict]:
    """
    Convert CSV files containing verification and validation data into dictionaries for V&V analysis.

    Parameters
    ----------
    v_and_v_data_files : list of str
        List of file paths to the CSV files containing verification and validation data.

    Returns
    -------
    list of dict
        A list containing two dictionaries: one for verification data and one for validation data.
        Each dictionary has keys corresponding to the legend keys extracted from the filenames, and values
        containing the mesh data and fluid state data extracted from the CSV files.
    """
    v_and_v_data = []
    for v_and_v_data_path in v_and_v_data_paths:
        df = pd.read_csv(v_and_v_data_path)
        v_and_v_data.append({
            df.columns[0]: df.iloc[1:, 0].values,
            df.columns[1]: df.iloc[1:, 1].values
        })

    return v_and_v_data


def _load_v_and_v_data(spec: Optional[VandVSpec], var_name: str):
    """Fetch parsed records and legend labels for one variable from an
    optional `VandVSpec`. Returns `([], ())` if there's nothing to plot.
    """
    if spec is None:
        return [], ()

    paths, labels = spec.for_variable(var_name)
    if not paths:
        return [], ()

    return _process_v_and_v_data(paths), labels


#: Axis-label formatting for each supported fluid state variable.
_AXIS_LABELS = {
    "Density": r"$\rho$ [kg/m³]",
    "Pressure": r"$p$ [Pa]",
    "Velocity": r"$u$ [m/s]",
    "internalEnergy": r"$e$ [J]",
    "Mach": r"$M$",
    "Entropy": r"$s$ [J/kg/K]",
    "Temperature": r"$T$ [K]",
}

#: Fluid state variables that can be read straight off the stored fluid
#: state, vs. ones that require deriving via the fluid model.
_DIRECT_FLUID_STATE_VARS = {"Density", "Pressure", "Velocity", "internalEnergy"}


def _compute_fluid_state_var_profile(fluidStateVarName: str, resultsAtIterIdx: dict) -> np.ndarray:
    """Extract or derive the requested fluid state variable's spatial
    profile (interior mesh nodes only) for a single simulation snapshot."""
    fluidState = resultsAtIterIdx["fluidState"]

    if fluidStateVarName in _DIRECT_FLUID_STATE_VARS:
        return fluidState[fluidStateVarName][1:-1]

    # Derived quantities require instantiating the driver/fluid model used
    # for that simulation.
    with pyshockflow.post_processing.HiddenPrints():
        driver = Driver(config=resultsAtIterIdx["config"])

    velocity = fluidState["Velocity"][1:-1]
    pressure = fluidState["Pressure"][1:-1]
    density = fluidState["Density"][1:-1]

    if fluidStateVarName == "Mach":
        return driver.fluidModel.computeMach_u_p_rho(velocity, pressure, density)
    if fluidStateVarName == "Entropy":
        return driver.fluidModel.computeEntropy_p_rho(pressure, density)
    if fluidStateVarName == "Temperature":
        return driver.fluidModel.computeTemperature_p_rho(pressure, density)

    raise ValueError(
        f"Unsupported fluidStateVarName '{fluidStateVarName}'. Supported "
        f"variables are: {sorted(_AXIS_LABELS)}."
    )



# -----------------------------
#  Main logic
# -----------------------------
def generate_fluid_state_var_profile_figs(
    resultsFolders: list[str],
    fluidStateVarNames: list[str],
    iterationIndexes: list[int],
    simulationLegendLabels: Optional[list[str]] = None,
    verificationData: Optional[VandVSpec] = None,
    validationData: Optional[VandVSpec] = None,
    showNozzleGeometry: bool = False,
) -> list[plt.Figure]:
    """Plot spatial profiles of one or more fluid state variables across one
    or more simulation results, optionally overlaid with verification and/or
    validation data.

    For each variable in `fluidStateVarNames`, a single figure is produced
    containing one line per entry in `simulationResultsHistories` (all such
    simulations must share the same nozzle geometry, since they are plotted
    on the same axes for comparison). Verification data is drawn as black
    'x' markers, validation data as black 'o' markers.

    Parameters
    ----------
    resultsFolders:
        List of paths to simulation results folders. 
    fluidStateVarNames:
        Fluid state variables to plot. One figure is generated per entry.
        Supported values: "Density", "Pressure", "Velocity",
        "internalEnergy", "Mach", "Entropy", "Temperature".
    iterationIndexes:
        Iteration index to plot for each entry in `simulationResultsHistories`
        (same order/length as `simulationResultsHistories`).
    simulationLegendLabels:
        Legend label for each entry in `simulationResultsHistories` (same
        order/length). Defaults to the dict keys of
        `simulationResultsHistories` if not given.
    verificationData, validationData:
        Optional `VandVSpec` instances describing reference data to overlay,
        keyed by the fluid state variable name they apply to. A variable
        in `fluidStateVarNames` with no matching entry in the spec is simply
        plotted without an overlay.
    showNozzleGeometry:
        If True, overlays a schematic of the nozzle geometry (scaled to the
        variable's y-range) on each figure.

    Returns
    -------
    list[matplotlib.figure.Figure]
        One figure per entry in `fluidStateVarNames`, in the same order.

    Raises
    ------
    ValueError
        If input lengths are inconsistent, if the simulations don't share a
        common nozzle geometry, or if provided verification/validation data
        does not actually contain the variable it's associated with.
    """
    # Extract simulationResultsHistories from resultsFolders
    simulationResultsHistories = collect_results_by_folder(resultsFolders)

    if simulationLegendLabels is None:
        simulationLegendLabels = list(simulationResultsHistories.keys())

    if len(iterationIndexes) != len(simulationResultsHistories):
        raise ValueError("iterationIndexes must be specified for all result folders.")
    if len(simulationLegendLabels) != len(simulationResultsHistories):
        raise ValueError("simulationLegendLabels must be specified for all result folders.")

    unsupported = set(fluidStateVarNames) - set(_AXIS_LABELS)
    if unsupported:
        raise ValueError(
            f"Unsupported fluidStateVarNames: {sorted(unsupported)}. "
            f"Supported variables are: {sorted(_AXIS_LABELS)}."
        )

    simulationResultsAtIterIdxs = extract_results_at_iteration_indexes(
        simulationResultsHistories, iterationIndexes
    )

    # All simulations being compared on one figure must share a nozzle geometry.
    nozzleGeometries = [
        resultsAtIterIdx["deviceGeometryData"]["deviceY"]
        for resultsAtIterIdx in simulationResultsAtIterIdxs.values()
    ]
    if not all(np.array_equal(nozzleGeometries[0], geom) for geom in nozzleGeometries):
        raise ValueError(
            "Multiple results folders were specified with differing nozzle "
            "geometries. generate_fluid_state_var_profile_figs() plots all "
            "simulations on shared axes, which requires identical nozzle "
            "geometries. Call it separately per geometry instead."
        )

    out_root = Path("Pictures")
    out_root.mkdir(parents=True, exist_ok=True)

    fluidStateVarProfileFigs = []

    for fluidStateVarName in fluidStateVarNames:
        fig, ax = plt.subplots(figsize=(12, 6))
        max_y = []

        verificationRecords, verificationLabels = _load_v_and_v_data(
            verificationData, fluidStateVarName
        )
        validationRecords, validationLabels = _load_v_and_v_data(
            validationData, fluidStateVarName
        )

        # Each V&V record's second key should describe the variable of interest.
        if not all(fluidStateVarName in list(record.keys())[1] for record in verificationRecords):
            raise ValueError(
                f"Verification data provided for '{fluidStateVarName}', but not all "
                f"verification data files contain this variable. Found variables: "
                f"{[list(record.keys())[1] for record in verificationRecords]}."
            )
        if not all(fluidStateVarName in list(record.keys())[1] for record in validationRecords):
            raise ValueError(
                f"Validation data provided for '{fluidStateVarName}', but not all "
                f"validation data files contain this variable. Found variables: "
                f"{[list(record.keys())[1] for record in validationRecords]}."
            )

        last_resultsFolder = None
        last_resultsAtIterIdx = None

        for simLabel, (resultsFolder, resultsAtIterIdx) in zip(
            simulationLegendLabels, simulationResultsAtIterIdxs.items()
        ):
            profile = _compute_fluid_state_var_profile(fluidStateVarName, resultsAtIterIdx)
            max_y.append(np.abs(profile).max())

            step = resultsAtIterIdx["iterIdx"]
            ax.plot(
                resultsAtIterIdx["meshData"]["xMeshNodes"][1:-1],
                profile,
                label=r"$%s: iteration=%s$" % (simLabel, step),
            )

            last_resultsFolder = resultsFolder
            last_resultsAtIterIdx = resultsAtIterIdx

        ax.set_ylabel(_AXIS_LABELS[fluidStateVarName])
        ax.set_xlabel(r"$x$ [m]")

        if showNozzleGeometry:
            max_y_val = max(max_y)
            deviceArea = last_resultsAtIterIdx["meshData"]["deviceAreaAtMeshNodes"][1:-1]
            ax.plot(
                last_resultsAtIterIdx["meshData"]["xMeshNodes"][1:-1],
                deviceArea * max_y_val * 1.2 * 0.3 / deviceArea.max(),
                label="Nozzle Geometry",
                color="gray",
                alpha=0.5,
                zorder=-1,
            )

        for record, label in zip(verificationRecords, verificationLabels):
            ax.scatter(
                record["deviceX"],
                record[list(record.keys())[1]],
                label=label,
                marker="x",
                color="black",
            )
        for record, label in zip(validationRecords, validationLabels):
            ax.scatter(
                record["deviceX"],
                record[list(record.keys())[1]],
                label=label,
                marker="o",
                color="black",
            )

        fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=6)
        fig.subplots_adjust(bottom=0.25)
        fig.savefig(out_root / f"{fluidStateVarName}.pdf", bbox_inches="tight")

        # Window positioning only works on some backends (e.g. TkAgg).
        try:
            manager = fig.canvas.manager
            manager.window.wm_geometry("+50+120")
            manager.set_window_title(f"{last_resultsFolder}: Simulation Results")
        except AttributeError:
            pass

        fluidStateVarProfileFigs.append(fig)

    return fluidStateVarProfileFigs








# ==========================================
#  Verification and Validation Metrics
# ==========================================
def compute_v_and_v_metrics(
    resultsFolders: list[str],
    fluidStateVarNames: list[str],
    verificationData: list[str] = None,
    validationData: list[str] = None,
) -> dict:
    """Compare each simulation result folder against its corresponding
    verification and/or validation data file, matched by position.
 
    Parameters
    ----------
    resultsFolders:
        Simulation result folders to compare, each at its final stored
        iteration.
    fluidStateVarNames:
        Fluid state variables that verification/validation data is allowed
        to be for; each supplied data file's variable is checked against
        this list.
    verificationData, validationData:
        Optional lists of file paths, one per entry in `resultsFolders`
        (same order, same length). At least one of the two must be given.
 
    Returns
    -------
    dict
        `{resultsFolder: {"verification": {...}, "validation": {...}}}`,
        with only the keys for which data was supplied. Each `{...}` holds
        "variable", "absolute_error" and "relative_error".
    """
    if verificationData is None and validationData is None:
        raise ValueError("At least one of verificationData or validationData must be provided.")
 
    if verificationData is not None and len(verificationData) != len(resultsFolders):
        raise ValueError(
            f"verificationData ({len(verificationData)} entries) must match "
            f"resultsFolders ({len(resultsFolders)} entries)."
        )
    if validationData is not None and len(validationData) != len(resultsFolders):
        raise ValueError(
            f"validationData ({len(validationData)} entries) must match "
            f"resultsFolders ({len(resultsFolders)} entries)."
        )
 
    simulationResultsHistories = collect_results_by_folder(resultsFolders)
    finalIterationIndexes = [
        history["iterIdxHistory"][-1] for history in simulationResultsHistories.values()
    ]
    resultsAtFinalIdxs = extract_results_at_iteration_indexes(
        simulationResultsHistories, finalIterationIndexes
    )
 
    table = Table(title="Verification and Validation")
    table.add_column("Results Folder", style="blue")
    table.add_column("Variable", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Max Absolute Error", justify="right", style="magenta")
    table.add_column("Max Relative Error", justify="right", style="green")
 
    v_and_v_metrics = {}
 
    for i, (resultsFolder, resultsAtIterIdx) in enumerate(resultsAtFinalIdxs.items()):
        xMeshNodes = resultsAtIterIdx["meshData"]["xMeshNodes"][1:-1]
        entry = {}
 
        for kind, dataList in (("verification", verificationData), ("validation", validationData)):
            if dataList is None:
                continue
 
            record = _process_v_and_v_data((dataList[i],))[0]
            var = list(record.keys())[1]
            if var not in fluidStateVarNames:
                raise ValueError(
                    f"{kind.capitalize()} data '{dataList[i]}' is for variable '{var}', "
                    f"which is not in fluidStateVarNames {fluidStateVarNames}."
                )
 
            simulationProfile = _compute_fluid_state_var_profile(var, resultsAtIterIdx)
            referenceX = record["deviceX"]
            referenceValue = record[var]
 
            if np.array_equal(referenceX, xMeshNodes):
                simulationAtReference = simulationProfile
            else:
                interpolant = interp1d(
                    xMeshNodes, simulationProfile, kind="linear", fill_value="extrapolate"
                )
                simulationAtReference = interpolant(referenceX)
 
            absolute_error = simulationAtReference - referenceValue
            relative_error = np.abs(absolute_error) / np.abs(referenceValue)
 
            entry[kind] = {
                "variable": var,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
            }
 
            table.add_row(
                str(resultsFolder),
                var,
                kind.capitalize(),
                f"{np.max(np.abs(absolute_error)):.3e}",
                f"{float(np.max(relative_error)*100):.1f} %",
            )
 
        v_and_v_metrics[resultsFolder] = entry
 
    Console().print(table)
 
    return v_and_v_metrics








# ==========================================
#  Expansion Path Thermoplot
# ==========================================
def generate_expansion_thermoplot(
    resultsFolders: list[str], 
    iterationIndexes: list[int],
    legend_labels: list[str],
    thermoplotConfigFilePath: str
    ) -> type[plt.Figure]:
    """
    Plot expansion paths from multiple simulation results on a single thermoplot.

    Args:
        thermoplotConfigFilePath: Path to the thermoplot configuration file.
        resultsFolders: List of paths to the results folders.
        iterationIndexes: List of iteration indexes to plot.
        legend_labels: List of labels for each expansion path.
        thermoplotConfigFilePath: Path to the thermoplot configuration file.
    """
    # Extract simulationResultsHistories from resultsFolders
    simulationResultsHistories = collect_results_by_folder(resultsFolders)

    # extact the iterIdx of interest from each of the results histories. and convert the 
    # collectdResultsHistories dict from a nested dict with subdicts according to the dicts
    # resulting form the assembleResultHistory() method to those resulting from the 
    # saveSingleIterResult() method. Instantiate a new dict with clearer name for this
    simulationResultsAtIterIdxs = extract_results_at_iteration_indexes(simulationResultsHistories, iterationIndexes)

    # if multiple results folders are specified, the results of each will be
    # plotted on the same figure for comparison. This only makes sense if the 
    # original simulations for which results are stored in the results folders
    # were performed with the same working fluid.
    workingFluids = []
    for resultsAtIterIdx in simulationResultsAtIterIdxs.values():
        workingFluids.append(resultsAtIterIdx["config"].fluidName())
        if not all(fluid == workingFluids[0] for fluid in workingFluids):
            raise ValueError("""
            Multiple results folders were specified. The thermoplot_expansion_plot()
            function is written to plot the results of multiple simulations on the same figure for comparison.
            For proper functionality, the working fluids of the simulations must be identical. 
            For simulations with different working fluids, please call the thermoplot_expansion_plot() 
            function separately for each results folder.
            """)

    # collect entropy and temperature arrays across all simulation files,
    # tracking global min/max to set thermoplot axis limits
    all_entropy = []
    all_temperature = []

    for resultsAtIterIdx in simulationResultsAtIterIdxs.values():
        with pyshockflow.post_processing.HiddenPrints():
            driver = Driver(config=resultsAtIterIdx["config"])

        entropy = driver.fluidModel.computeEntropy_p_rho(
            resultsAtIterIdx["fluidState"]['Pressure'][1:-1],
            resultsAtIterIdx["fluidState"]['Density'][1:-1]
        )
        temperature = driver.fluidModel.computeTemperature_p_rho(
            resultsAtIterIdx["fluidState"]['Pressure'][1:-1],
            resultsAtIterIdx["fluidState"]['Density'][1:-1]
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

    # working fluid can be obtained from any of the config files
    # and should match with that specified in the thermoplot config file.
    config = next(iter(simulationResultsAtIterIdxs.values()))["config"]
    config_thermoplot = ConfigThermoplot(config_file=thermoplotConfigFilePath)
    config_thermoplot.get_thermoplot_settings()
    if config.fluidName() != config_thermoplot.thermoplot_settings["fluid_name"]:
        raise ValueError(f"""
            Working fluid in simulation results ({config.fluidName()}) does not 
            match working fluid in thermoplot config file ({config.thermoplot_settings['fluid_name']}).
            Please ensure that the thermoplot config file is set up for the correct working fluid for 
            the simulation results you are trying to analyze.
            """)

    # get plot background
    fig = thermoplot_cached(thermoplotConfigFilePath, thermoplot_overwrite_settings=thermoplot_overwrite_settings)
    ax = fig.get_axes()[0]

    # plot each expansion path with a distinct colour
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (entropy, temperature, label) in enumerate(zip(all_entropy, all_temperature, legend_labels)):
        color = color_cycle[i % len(color_cycle)]
        ax.plot(entropy, temperature, color=color, marker='o', markersize=2, label=label)

    ax.legend()

    return fig




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





# ========================================================
#  Legacy
# ========================================================
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


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
