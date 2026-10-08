from __future__ import annotations

import pickle 
import sys
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

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
#  Helpers
# ================================
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


        
def collect_results_by_folder(
    resultsFolders: list[str],
    iterationIndexes: list[int] | None = None,
) -> dict:
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

    if iterationIndexes is not None and len(iterationIndexes) != len(resultsFolders):
        raise ValueError(
            "iterationIndexes must contain one index for each results folder."
        )

    for folderIndex, resultsFolder in enumerate(resultsFolders):
        # check if results files have already been grouped
        if Path(resultsFolder / "Results.pik").is_file():
            with open(str(Path(resultsFolder / "Results.pik")), 'rb') as file:
                simulationResultsHisotry = pickle.load(file)
        elif iterationIndexes is not None:
            iterIdx = iterationIndexes[folderIndex]
            available = sorted(int(f.stem.split("_")[1]) for f in resultsFolder.glob("step_*.pik"))
            if iterIdx == -1:
                iterIdx = available[-1]
            stepPath = resultsFolder / f"step_{iterIdx:06d}.pik"
            if not stepPath.is_file():
                available = sorted(int(f.stem.split("_")[1]) for f in resultsFolder.glob("step_*.pik"))
                raise ValueError(
                    f"\n"
                    f"{resultsFolder}: No results available for iteration index {iterIdx}.\n"
                    f"Available iteration indexes ({len(available)} total):\n"
                    f"{np.array2string(np.asarray(available), threshold=15, edgeitems=7)}\n"
                )
            with tempfile.TemporaryDirectory() as temporaryDirectory:
                temporaryStepPath = Path(temporaryDirectory) / stepPath.name
                shutil.copy2(stepPath, temporaryStepPath)
                simulationResultsHisotry = Driver.assembleResultHistory(
                    Path(temporaryDirectory), verbose=False
                )
        else:
            # if not, apply driver assembleResultHistory() method to the unfinished folder
            # to create a dictionary of similar structure to the grouped results file.
            # this will simplify future operations. 
            simulationResultsHisotry = Driver.assembleResultHistory(resultsFolder, verbose=False)
        simulationResultsHistories[str(resultsFolder)] = simulationResultsHisotry

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
        if Path(Path(resultsFolder) / "Results.pik").is_file():
            iterIdxHistory = iterResultsHistory["iterIdxHistory"]
            if iterIdx == -1:
                iterIdx = iterIdxHistory[-1]
        else:
            available = sorted(int(f.stem.split("_")[1]) for f in Path(resultsFolder).glob("step_*.pik"))
            if iterIdx == -1:
                iterIdx = available[-1]
        if iterIdx not in iterResultsHistory["iterIdxHistory"]:
            raise ValueError(f"{resultsFolder}: No results available for iteration index {iterIdx}. Available iteration indexes are: {np.array2string(np.asarray(iterResultsHistory['iterIdxHistory']), threshold=15, edgeitems=7)}")

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
def expansion_device_geometry_plot(config: Config = None, resultsFolder: dict = None, showGhostNodes: bool = False) -> None:
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

        if not showGhostNodes:
            meshX = meshX[1:-1]

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


def _compute_fluid_state_var_profile(
    fluidStateVarName: str,
    resultsAtIterIdx: dict,
    showGhostNodes: bool = False,
) -> np.ndarray:
    """Extract or derive the requested fluid state variable's spatial
    profile (interior mesh nodes only) for a single simulation snapshot."""
    fluidState = resultsAtIterIdx["fluidState"]

    node_slice = slice(None) if showGhostNodes else slice(1, -1)

    if fluidStateVarName in _DIRECT_FLUID_STATE_VARS:
        return fluidState[fluidStateVarName][node_slice]

    # Derived quantities require instantiating the driver/fluid model used
    # for that simulation.
    with pyshockflow.post_processing.HiddenPrints():
        driver = Driver(config=resultsAtIterIdx["config"])

    velocity = fluidState["Velocity"][node_slice]
    pressure = fluidState["Pressure"][node_slice]
    density = fluidState["Density"][node_slice]

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
    showGhostNodes: bool = False,
    separatePlots: bool = False,
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
    # Load only the requested step files when the simulation is unfinished.
    simulationResultsHistories = collect_results_by_folder(
        resultsFolders, iterationIndexes=iterationIndexes
    )

    if simulationLegendLabels is None:
        simulationLegendLabels = list(simulationResultsHistories.keys())

    if len(simulationLegendLabels) < len(simulationResultsHistories):
        raise ValueError("simulationLegendLabels must be specified for all result folders. These must match one-to-one in order.")
    elif len(simulationLegendLabels) > len(simulationResultsHistories):
        raise ValueError("Too many simulationLegendLabels specified for the number of result folders. These must match one-to-one in order.")

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
    if not separatePlots and not all(
        np.array_equal(nozzleGeometries[0], geom) for geom in nozzleGeometries
    ):
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
        simulation_items = list(simulationResultsAtIterIdxs.items())
        plot_items = (
            [simulation_items]
            if not separatePlots
            else [[simulation_item] for simulation_item in simulation_items]
        )

        for plot_index, selected_items in enumerate(plot_items):
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
            plot_labels = (
                simulationLegendLabels
                if not separatePlots
                else [simulationLegendLabels[plot_index]]
            )

            for simLabel, (resultsFolder, resultsAtIterIdx) in zip(
                plot_labels, selected_items
            ):
                profile = _compute_fluid_state_var_profile(
                    fluidStateVarName, resultsAtIterIdx, showGhostNodes=showGhostNodes
                )
                max_y.append(np.abs(profile).max())

                step = resultsAtIterIdx["iterIdx"]
                ax.plot(
                    resultsAtIterIdx["meshData"]["xMeshNodes"]
                    if showGhostNodes
                    else resultsAtIterIdx["meshData"]["xMeshNodes"][1:-1],
                    profile,
                    label=f"{simLabel}: iteration={step}",
                )

                last_resultsFolder = resultsFolder
                last_resultsAtIterIdx = resultsAtIterIdx

            ax.set_ylabel(_AXIS_LABELS[fluidStateVarName])
            ax.set_xlabel(r"$x$ [m]")

            if showNozzleGeometry:
                max_y_val = max(max_y)
                node_slice = slice(None) if showGhostNodes else slice(1, -1)
                deviceArea = last_resultsAtIterIdx["meshData"]["deviceAreaAtMeshNodes"][node_slice]
                ax.plot(
                    last_resultsAtIterIdx["meshData"]["xMeshNodes"][node_slice],
                    deviceArea * max_y_val * 1.2 * 0.3 / deviceArea.max(),
                    label="Nozzle Geometry",
                    color="gray",
                    alpha=0.5,
                    zorder=-1,
                )

            verification_items = (
                zip(verificationRecords, verificationLabels)
                if not separatePlots
                else zip(
                    verificationRecords[plot_index:plot_index + 1],
                    verificationLabels[plot_index:plot_index + 1],
                )
            )
            for record, label in verification_items:
                ax.scatter(
                    record["deviceX"],
                    record[list(record.keys())[1]],
                    label=label,
                    marker="x",
                    color="black",
                )
            validation_items = (
                zip(validationRecords, validationLabels)
                if not separatePlots
                else zip(
                    validationRecords[plot_index:plot_index + 1],
                    validationLabels[plot_index:plot_index + 1],
                )
            )
            for record, label in validation_items:
                ax.scatter(
                    record["deviceX"],
                    record[list(record.keys())[1]],
                    label=label,
                    marker="o",
                    color="black",
                )

            fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=12)
            fig.subplots_adjust(bottom=0.25)
            suffix = f"_{plot_index}" if separatePlots else ""
            fig.savefig(out_root / f"{fluidStateVarName}{suffix}.pdf", bbox_inches="tight")

            # Window positioning only works on some backends (e.g. TkAgg).
            try:
                manager = fig.canvas.manager
                manager.window.wm_geometry("+50+120")
                manager.set_window_title(f"{last_resultsFolder}: Simulation Results")
            except AttributeError:
                pass

            fluidStateVarProfileFigs.append(fig)

    return fluidStateVarProfileFigs






# ====================
#  Animation
# ====================
"""Smooth MP4 animation of PyShockFlow profiles (pre-rendered, hardware-decoded).

pip install imageio-ffmpeg   # bundles an ffmpeg binary, no system install needed
"""
import contextlib
import io
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter

from pyshockflow.driver import Driver
from pyshockflow.post_processing import collect_results_by_folder

# NOTE: deliberately NO matplotlib.use("Agg") here.
# Setting the backend at import time breaks interactive plotting
# (fig.show / plt.show) for the rest of the module.

_STATE_FIELDS = (
    ("Density", "Density", "kg/m^3"),
    ("Pressure", "Pressure", "Pa"),
    ("Velocity", "Velocity", "m/s"),
    ("Internal energy", "staticInternalEnergy", "J/kg"),
    ("Temperature", "Temperature", "K"),
    ("Entropy", "Entropy", "J/(kg K)"),
    ("Mach", "Mach", "-"),
)

_PROFILE_LABELS = {
    "Density": ("Density", "kg/m^3"),
    "Pressure": ("Pressure", "Pa"),
    "Velocity": ("Velocity", "m/s"),
    "staticInternalEnergy": ("Internal energy", "J/kg"),
    "Temperature": ("Temperature", "K"),
    "Entropy": ("Entropy", "J/(kg K)"),
    "Mach": ("Mach", "-"),
}

_CONVERGENCE_VARIABLES = ("Density", "Velocity", "Pressure", "staticInternalEnergy")


def _profile_key(variable: str) -> str:
    return "staticInternalEnergy" if variable == "internalEnergy" else variable


def _node_slice(show_ghost_nodes: bool) -> slice:
    return slice(None) if show_ghost_nodes else slice(1, -1)


def _unconverged_masks(
    history: dict,
    config,
    show_ghost_nodes: bool,
) -> np.ndarray:
    """Convergence test for every stored frame at once.

    Returns a boolean array of shape (nodes, frames).  Entry [i, k] is True
    when node i failed the configured convergence test in any of the last
    ``convergencePatience()`` transitions between stored snapshots ending at
    frame k.  Frames with fewer than ``patience`` preceding transitions are
    treated as unconverged, because the requested patience cannot be
    established.
    """
    patience = config.convergencePatience()
    tolerance = config.convergenceTolerance()
    fluid_history = history["fluidStateHistory"]
    n_nodes, n_frames = np.asarray(fluid_history["Density"]).shape

    changed = np.zeros((n_nodes, max(n_frames - 1, 0)), dtype=bool)
    for name in _CONVERGENCE_VARIABLES:
        values = np.asarray(fluid_history[name])
        scale = np.max(np.abs(values[:, :-1]), axis=0) + 1e-300
        changed |= np.abs(np.diff(values, axis=1)) / scale >= tolerance

    csum = np.concatenate(
        [np.zeros((n_nodes, 1), dtype=np.int32), np.cumsum(changed, axis=1, dtype=np.int32)],
        axis=1,
    )
    masks = np.ones((n_nodes, n_frames), dtype=bool)
    if n_frames > patience:
        masks[:, patience:] = (csum[:, patience:] - csum[:, : n_frames - patience]) > 0

    return masks[_node_slice(show_ghost_nodes)]


def _make_video_writer(output_path: str, fps: int):
    """Return (writer, final_path). Prefers MP4, falls back to GIF."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = None
    if ffmpeg is not None:
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
        writer = FFMpegWriter(
            fps=fps, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "20",
                        "-preset", "veryfast", "-movflags", "+faststart"],
        )
        return writer, output_path
    print("ffmpeg not found; falling back to GIF (larger and lower quality).")
    return PillowWriter(fps=min(fps, 30)), str(Path(output_path).with_suffix(".gif"))


def render_profile_video(
    results_folder: str,
    fluid_state_var_name: str,
    output_path: str = "profile.mp4",
    speed: float = 1.0,          # < 1 slows the video (frames are repeated);
                                 # > 1 skips frames (faster playback)
    fps: int = 60,               # playback frame rate of the video
    show_ghost_nodes: bool = False,
    show_unconverged_nodes: bool = False,
    size_px: tuple[int, int] = (1280, 720),
    write_html: bool = True,
) -> str:
    if fluid_state_var_name not in _PROFILE_LABELS:
        raise ValueError(f"Unsupported variable: {sorted(_PROFILE_LABELS)}")
    if speed <= 0:
        raise ValueError("speed must be a positive number")

    history = next(iter(collect_results_by_folder([results_folder]).values()))
    iters = np.asarray(history["iterIdxHistory"])
    times = np.asarray(history["timeHistory"])
    config = history["config"]
    nodes = _node_slice(show_ghost_nodes)
    x = np.asarray(history["meshData"]["xMeshNodes"])[nodes]

    # --- batched state computation (one fluid-model call per property) ---
    fh = history["fluidStateHistory"]
    key = _profile_key(fluid_state_var_name)
    if key in ("Temperature", "Entropy", "Mach"):
        p, rho, u = (np.asarray(fh[n])[nodes] for n in ("Pressure", "Density", "Velocity"))
        with contextlib.redirect_stdout(io.StringIO()):
            fm = Driver(config=config).fluidModel
        shape = p.shape
        with np.errstate(invalid="ignore", divide="ignore"):
            if key == "Temperature":
                prof = fm.computeTemperature_p_rho(p.ravel(), rho.ravel())
            elif key == "Entropy":
                prof = fm.computeEntropy_p_rho(p.ravel(), rho.ravel())
            else:
                prof = fm.computeMach_u_p_rho(u.ravel(), p.ravel(), rho.ravel())
        profiles = np.asarray(prof).reshape(shape)
    else:
        profiles = np.asarray(fh[key])[nodes]

    masks = _unconverged_masks(history, config, show_ghost_nodes) if show_unconverged_nodes else None

    # Frame selection:
    #   speed >= 1  → take every int(speed)-th frame
    #   speed <  1  → keep every frame and repeat it so the clip is longer
    if speed >= 1:
        frames = np.arange(0, iters.size, int(round(speed)))
    else:
        repeats = max(1, int(round(1.0 / speed)))
        frames = np.repeat(np.arange(iters.size), repeats)

    finite = profiles[np.isfinite(profiles)]
    lo, hi = float(finite.min()), float(finite.max())
    pad = 0.05 * (hi - lo) if hi > lo else max(abs(lo) * 0.05, 1.0)

    # --- static figure, built once; only artists' data change per frame ---
    label, unit = _PROFILE_LABELS[fluid_state_var_name]
    dpi = 100
    fig, ax = plt.subplots(figsize=(size_px[0] / dpi, size_px[1] / dpi), dpi=dpi)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("x [m]")
    ax.set_ylabel(f"{label} [{unit}]" if unit != "-" else label)
    ax.grid(alpha=0.3)
    (line,) = ax.plot(x, profiles[:, frames[0]], lw=1.2, color="tab:blue")
    pts = ax.scatter(x, profiles[:, frames[0]], s=10, zorder=3, color="green")
    title = ax.set_title("")
    fig.tight_layout()

    def update(k: int):
        y = profiles[:, k]
        line.set_ydata(y)
        pts.set_offsets(np.column_stack([x, y]))
        if masks is not None:
            pts.set_color(np.where(masks[:, k], "red", "green"))
        title.set_text(f"{label}: iteration={iters[k]}, time={times[k]:.6g} s")

    writer, output_path = _make_video_writer(output_path, fps)
    with writer.saving(fig, output_path, dpi):
        for k in frames:
            update(k)
            writer.grab_frame()
    plt.close(fig)

    if write_html:
        html = Path(output_path).with_suffix(".html")
        html.write_text(f"""<!doctype html><meta charset="utf-8">
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center">
<video id="v" src="{Path(output_path).name}" controls loop autoplay muted
       style="max-width:100%;max-height:90vh"></video><br>
{''.join(f'<button onclick="v.playbackRate={s}">{s}x</button> ' for s in (0.25, 0.5, 1, 2, 4))}
</body>""")
    return output_path







# ==========================================
#  Verification and Validation Metrics
# ==========================================
def compute_v_and_v_metrics(
    resultsFolders: list[str],
    fluidStateVarNames: list[str],
    verificationData: VandVSpec = None,
    validationData: VandVSpec = None,
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
        Optional `VandVSpec` objects. Their files are flattened in the
        declared variable/path order and matched one-to-one with
        `resultsFolders`. At least one of the two must be given.
 
    Returns
    -------
    dict
        `{resultsFolder: {"verification": {...}, "validation": {...}}}`,
        with only the keys for which data was supplied. Each `{...}` holds
        "variable", "absolute_error" and "relative_error".
    """
    if verificationData is None and validationData is None:
        raise ValueError("At least one of verificationData or validationData must be provided.")

    def _records_in_order(spec: Optional[VandVSpec]) -> list[dict]:
        if spec is None:
            return []
        records = []
        for variable, paths in spec.paths.items():
            records.extend(_process_v_and_v_data(paths))
        return records

    verificationRecords = _records_in_order(verificationData)
    validationRecords = _records_in_order(validationData)
    if verificationData is not None and len(verificationRecords) != len(resultsFolders):
        raise ValueError(
            f"verificationData ({len(verificationRecords)} entries) must match "
            f"resultsFolders ({len(resultsFolders)} entries)."
        )
    if validationData is not None and len(validationRecords) != len(resultsFolders):
        raise ValueError(
            f"validationData ({len(validationRecords)} entries) must match "
            f"resultsFolders ({len(resultsFolders)} entries)."
        )

    simulationResultsHistories = collect_results_by_folder(resultsFolders, iterationIndexes=[-1]*len(resultsFolders))
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
 
        for kind, records in (("verification", verificationRecords), ("validation", validationRecords)):
            if not records:
                continue

            record = records[i]
            var = list(record.keys())[1]
            if var not in fluidStateVarNames:
                raise ValueError(
                    f"{kind.capitalize()} data for result '{resultsFolders[i]}' is "
                    f"for variable '{var}', "
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
    simulationLegendLabels: list[str],
    thermoplotConfigFilePath: str,
    showGhostNodes: bool = False,
    separatePlots: bool = False,
    ) -> type[plt.Figure]:
    """
    Plot expansion paths from multiple simulation results on a single thermoplot.

    Args:
        thermoplotConfigFilePath: Path to the thermoplot configuration file.
        resultsFolders: List of paths to the results folders.
        iterationIndexes: List of iteration indexes to plot.
        simulationLegendLabels: List of labels for each expansion path.
        thermoplotConfigFilePath: Path to the thermoplot configuration file.
    """
    # Extract simulationResultsHistories from resultsFolders
    simulationResultsHistories = collect_results_by_folder(
        resultsFolders, iterationIndexes=iterationIndexes
    )

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

        node_slice = slice(None) if showGhostNodes else slice(1, -1)
        entropy = driver.fluidModel.computeEntropy_p_rho(
            resultsAtIterIdx["fluidState"]['Pressure'][node_slice],
            resultsAtIterIdx["fluidState"]['Density'][node_slice]
        )
        temperature = driver.fluidModel.computeTemperature_p_rho(
            resultsAtIterIdx["fluidState"]['Pressure'][node_slice],
            resultsAtIterIdx["fluidState"]['Density'][node_slice]
        )

        all_entropy.append(entropy)
        all_temperature.append(temperature)

    # if nan or inf in any of the entropy or temperature arrays, remove these from the array
    for i in range(len(all_entropy)):
        all_entropy[i] = all_entropy[i][~np.isnan(all_entropy[i])]
        all_entropy[i] = all_entropy[i][~np.isinf(all_entropy[i])]
        all_temperature[i] = all_temperature[i][~np.isnan(all_temperature[i])]
        all_temperature[i] = all_temperature[i][~np.isinf(all_temperature[i])]

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

    # get plot background and plot each expansion path
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    figures = []
    plot_items = zip(all_entropy, all_temperature, simulationLegendLabels)
    for i, (entropy, temperature, label) in enumerate(plot_items):
        fig = thermoplot_cached(
            thermoplotConfigFilePath,
            thermoplot_overwrite_settings=thermoplot_overwrite_settings,
        )
        ax = fig.get_axes()[0]
        color = color_cycle[i % len(color_cycle)]
        ax.plot(entropy, temperature, color=color, marker='o', markersize=2, label=label)
        ax.legend()
        try:
            manager = fig.canvas.manager
            manager.window.wm_geometry("+50+120")
            window_name = resultsFolders[i] if separatePlots else "Multiple simulations"
            manager.set_window_title(f"{window_name}: Expansion Path")
        except AttributeError:
            pass
        figures.append(fig)
        if not separatePlots:
            break

    return figures if separatePlots else figures[0]








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



