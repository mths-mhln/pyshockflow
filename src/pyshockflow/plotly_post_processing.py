"""Interactive Plotly visualizations for PyShockFlow results.

The existing Matplotlib plotting functions remain unchanged.  These helpers
reuse their result extraction and fluid-property calculations while adding
complete SI-unit state information to Plotly hover labels.

Efficiency notes
----------------
* The fluid model is built once per result set instead of once per snapshot
  (constructing a ``Driver`` is by far the most expensive step).
* Fluid-property calls are batched over all stored frames in the animation.
* The convergence test is evaluated for all frames at once with a vectorized
  rolling window.
* Animation frames carry only the data that changes (y, customdata, marker
  colors); static parts (x, name, hovertemplate) live in the initial trace.
"""

from __future__ import annotations

import contextlib
import io
import sys
from typing import Optional

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly

from pyshockflow.driver import Driver
from pyshockflow.post_processing import (
    VandVSpec,
    _load_v_and_v_data,
    collect_results_by_folder,
    extract_results_at_iteration_indexes,
)
from thermoplot.configthermoplot import ConfigThermoplot
from thermoplot.thermoplot import thermoplot_cached


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


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _node_slice(show_ghost_nodes: bool) -> slice:
    return slice(None) if show_ghost_nodes else slice(1, -1)


def _profile_key(variable: str) -> str:
    return "staticInternalEnergy" if variable == "internalEnergy" else variable


def _fluid_model(config):
    """Build the fluid model once, silencing Driver construction output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return Driver(config=config).fluidModel


def _state_values(
    results_at_iteration: dict,
    show_ghost_nodes: bool,
    fluid_model=None,
) -> dict[str, np.ndarray]:
    """Return native and derived state fields on the requested mesh nodes.

    Pass ``fluid_model`` to avoid rebuilding the Driver on every call.
    """
    nodes = _node_slice(show_ghost_nodes)
    fluid_state = results_at_iteration["fluidState"]
    if fluid_model is None:
        fluid_model = _fluid_model(results_at_iteration["config"])

    pressure = np.asarray(fluid_state["Pressure"])[nodes]
    density = np.asarray(fluid_state["Density"])[nodes]
    velocity = np.asarray(fluid_state["Velocity"])[nodes]

    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "Density": density,
            "Pressure": pressure,
            "Velocity": velocity,
            "staticInternalEnergy": np.asarray(fluid_state["staticInternalEnergy"])[nodes],
            "Temperature": np.asarray(fluid_model.computeTemperature_p_rho(pressure, density)),
            "Entropy": np.asarray(fluid_model.computeEntropy_p_rho(pressure, density)),
            "Mach": np.asarray(fluid_model.computeMach_u_p_rho(velocity, pressure, density)),
        }


def _hover_customdata(
    state: dict[str, np.ndarray],
    mesh_node_numbers: Optional[np.ndarray] = None,
) -> np.ndarray:
    columns = [state[key] for _, key, _ in _STATE_FIELDS]
    if mesh_node_numbers is not None:
        columns.insert(0, mesh_node_numbers)
    return np.column_stack(columns)


def _hover_template(include_mesh_node: bool = False) -> str:
    lines = ["<b>%{fullData.name}</b>"]
    if include_mesh_node:
        lines.append("Mesh node: %{customdata[0]}")
        customdata_offset = 1
    else:
        customdata_offset = 0
    lines.append("x: %{x:.6g} m")
    for index, (label, _, unit) in enumerate(_STATE_FIELDS):
        unit_suffix = f" {unit}" if unit != "-" else ""
        lines.append(
            f"{label}: %{{customdata[{index + customdata_offset}]:.6g}}{unit_suffix}"
        )
    lines.append("<extra></extra>")
    return "<br>".join(lines)


def _reference_hover_template(label: str, variable_label: str, unit: str) -> str:
    unit_suffix = f" {unit}" if unit != "-" else ""
    return (
        f"<b>{label}</b><br>"
        "x: %{x:.6g} m<br>"
        f"{variable_label}: %{{y:.6g}}{unit_suffix}"
        "<extra></extra>"
    )


def _frame_index(history: dict, iteration_index: int) -> int:
    """Position of ``iteration_index`` in the stored snapshot history."""
    matches = np.flatnonzero(np.asarray(history["iterIdxHistory"]) == iteration_index)
    if matches.size == 0:
        raise ValueError(
            f"Iteration {iteration_index} is not available in the result history."
        )
    return int(matches[0])


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

    # changed[:, t] is True if the transition from frame t to t+1 exceeds tol.
    changed = np.zeros((n_nodes, max(n_frames - 1, 0)), dtype=bool)
    for name in _CONVERGENCE_VARIABLES:
        values = np.asarray(fluid_history[name])
        scale = np.max(np.abs(values[:, :-1]), axis=0) + 1e-300
        changed |= np.abs(np.diff(values, axis=1)) / scale >= tolerance

    # Rolling window sum over the last `patience` transitions via cumsum.
    csum = np.concatenate(
        [np.zeros((n_nodes, 1), dtype=np.int32), np.cumsum(changed, axis=1, dtype=np.int32)],
        axis=1,
    )
    masks = np.ones((n_nodes, n_frames), dtype=bool)
    if n_frames > patience:
        masks[:, patience:] = (csum[:, patience:] - csum[:, : n_frames - patience]) > 0

    return masks[_node_slice(show_ghost_nodes)]


# --------------------------------------------------------------------------
# Static profiles
# --------------------------------------------------------------------------

def generate_fluid_state_var_profile_plotly(
    results_folders: list[str],
    fluid_state_var_names: list[str],
    iteration_indexes: list[int],
    simulation_legend_labels: Optional[list[str]] = None,
    verification_data: Optional[VandVSpec] = None,
    validation_data: Optional[VandVSpec] = None,
    show_nozzle_geometry: bool = False,
    show_ghost_nodes: bool = False,
    show_unconverged_nodes: bool = False,
) -> list[go.Figure]:
    """Create interactive spatial profiles with complete state hover labels.

    One Plotly figure is returned for each requested profile.  Hovering a
    simulation point shows density, pressure, velocity, internal energy,
    temperature, entropy, and Mach number, all in SI units. Optional
    verification and validation records use the same `VandVSpec` and CSV
    parsing as the Matplotlib profile plots. When `show_unconverged_nodes` is
    true, simulation markers are colored red for nodes that fail the configured
    convergence test during the recent patience window and green otherwise.
    """
    unknown = set(fluid_state_var_names) - set(_PROFILE_LABELS)
    if unknown:
        raise ValueError(f"Unsupported fluid state variables: {sorted(unknown)}")

    histories = collect_results_by_folder(
        results_folders,
        None if show_unconverged_nodes else iteration_indexes,
    )
    results = extract_results_at_iteration_indexes(histories, iteration_indexes)
    if simulation_legend_labels is None:
        simulation_legend_labels = list(results)
    if len(simulation_legend_labels) != len(results):
        raise ValueError("simulation_legend_labels must match results_folders")

    nodes = _node_slice(show_ghost_nodes)
    hover_template = _hover_template(include_mesh_node=True)

    # Everything below depends only on the result, not on the plotted variable,
    # so compute it once and reuse it for every figure.
    per_result = {}
    for results_folder, result in results.items():
        state = _state_values(result, show_ghost_nodes)
        mesh_nodes = np.asarray(result["meshData"]["xMeshNodes"])
        entry = {
            "state": state,
            "x": mesh_nodes[nodes],
            "customdata": _hover_customdata(state, np.arange(mesh_nodes.size)[nodes]),
            "marker": None,
        }
        if show_unconverged_nodes:
            history = histories[results_folder]
            masks = _unconverged_masks(history, result["config"], show_ghost_nodes)
            unconverged = masks[:, _frame_index(history, result["iterIdx"])]
            entry["marker"] = {"color": np.where(unconverged, "red", "green")}
        per_result[results_folder] = entry

    figures = []
    for variable in fluid_state_var_names:
        verification_records, verification_labels = _load_v_and_v_data(
            verification_data, variable
        )
        validation_records, validation_labels = _load_v_and_v_data(
            validation_data, variable
        )

        for data_kind, records in (
            ("Verification", verification_records),
            ("Validation", validation_records),
        ):
            invalid_records = [
                list(record.keys())[1]
                for record in records
                if variable not in list(record.keys())[1]
            ]
            if invalid_records:
                raise ValueError(
                    f"{data_kind} data provided for '{variable}', but not all "
                    f"data files contain this variable. Found variables: "
                    f"{invalid_records}."
                )

        variable_label, unit = _PROFILE_LABELS[variable]
        key = _profile_key(variable)
        figure = go.Figure()
        last_result = None
        max_profile = 0.0

        for label, (results_folder, result) in zip(
            simulation_legend_labels, results.items()
        ):
            entry = per_result[results_folder]
            profile = entry["state"][key]
            max_profile = max(max_profile, float(np.nanmax(np.abs(profile))))
            figure.add_trace(go.Scatter(
                x=entry["x"],
                y=profile,
                mode="lines+markers",
                name=f"{label}: iteration={result['iterIdx']}",
                customdata=entry["customdata"],
                hovertemplate=hover_template,
                marker=entry["marker"],
            ))
            last_result = result

        for record, label in zip(verification_records, verification_labels):
            variable_key = list(record.keys())[1]
            figure.add_trace(go.Scatter(
                x=record["deviceX"],
                y=record[variable_key],
                mode="markers",
                name=label,
                marker={"symbol": "x", "color": "black"},
                hovertemplate=_reference_hover_template(label, variable_label, unit),
            ))
        for record, label in zip(validation_records, validation_labels):
            variable_key = list(record.keys())[1]
            figure.add_trace(go.Scatter(
                x=record["deviceX"],
                y=record[variable_key],
                mode="markers",
                name=label,
                marker={"symbol": "circle", "color": "black"},
                hovertemplate=_reference_hover_template(label, variable_label, unit),
            ))

        if show_nozzle_geometry and last_result is not None:
            area = np.asarray(last_result["meshData"]["deviceAreaAtMeshNodes"])[nodes]
            figure.add_trace(go.Scatter(
                x=np.asarray(last_result["meshData"]["xMeshNodes"])[nodes],
                y=area * max_profile * 1.2 * 0.3 / np.nanmax(area),
                mode="lines",
                name="Nozzle geometry",
                line={"color": "gray"},
                hoverinfo="skip",
            ))

        figure.update_layout(
            template="plotly_white",
            xaxis_title="x [m]",
            yaxis_title=f"{variable_label} [{unit}]" if unit != "-" else variable_label,
            hovermode="closest",
        )
        figures.append(figure)

    return figures


# --------------------------------------------------------------------------
# Animation
# --------------------------------------------------------------------------

def generate_fluid_state_var_profile_animation_plotly(
    results_folder: str,
    fluid_state_var_name: str,
    show_ghost_nodes: bool = False,
    interval: int = 100,
    show_unconverged_nodes: bool = False,
    compact_hover_data: bool = True,
) -> go.Figure:
    """Create a browser-based animated fluid-state profile.

    The returned Plotly figure includes an iteration slider, play controls,
    and playback buttons for half, normal, and double speed. Hover labels
    include the mesh-node number and all available state variables. When
    `show_unconverged_nodes` is true, each frame colors nodes red when they
    fail the configured convergence test during the recent patience window,
    and green otherwise.

    `interval` is the requested difference between frame iteration indexes.
    It must be a multiple of the result write interval; values below the write
    interval use the write interval instead.

    `compact_hover_data` stores hover values as float32, roughly halving the
    size of the figure. Precision (~7 digits) exceeds the ``.6g`` shown in the
    hover label.
    """
    if fluid_state_var_name not in _PROFILE_LABELS:
        raise ValueError(
            f"Unsupported fluid state variable: {fluid_state_var_name}. "
            f"Supported variables are: {sorted(_PROFILE_LABELS)}"
        )
    if interval < 0:
        raise ValueError("interval must be non-negative.")

    histories = collect_results_by_folder([results_folder])
    history = next(iter(histories.values()))
    iteration_indexes = np.asarray(history["iterIdxHistory"])
    if iteration_indexes.size == 0:
        raise ValueError(f"No stored iterations found in '{results_folder}'.")
    if iteration_indexes.size < 2:
        raise ValueError(
            "At least two stored iterations are required to infer the original writeInterval."
        )

    write_interval = int(iteration_indexes[1] - iteration_indexes[0])
    if write_interval <= 0:
        raise ValueError("Saved iteration indexes must be strictly increasing.")
    if interval % write_interval != 0:
        raise ValueError(
            f"The specified interval ({interval}) is not a multiple of the original "
            f"writeInterval ({write_interval})."
        )
    frame_interval = max(interval, write_interval)
    frame_indexes = np.flatnonzero(
        (iteration_indexes - iteration_indexes[0]) % frame_interval == 0
    )

    config = history["config"]
    nodes = _node_slice(show_ghost_nodes)
    mesh_nodes = np.asarray(history["meshData"]["xMeshNodes"])
    x_values = mesh_nodes[nodes]
    mesh_node_numbers = np.arange(mesh_nodes.size)[nodes]

    # --- State for all frames, with one batched fluid-model call per property.
    fluid_history = history["fluidStateHistory"]
    pressure, density, velocity, energy = (
        np.asarray(fluid_history[name])[nodes]
        for name in ("Pressure", "Density", "Velocity", "staticInternalEnergy")
    )
    shape = pressure.shape  # (nodes, frames)
    p_flat, rho_flat, u_flat = pressure.ravel(), density.ravel(), velocity.ravel()

    fluid_model = _fluid_model(config)
    with np.errstate(invalid="ignore", divide="ignore"):
        temperature = np.asarray(fluid_model.computeTemperature_p_rho(p_flat, rho_flat)).reshape(shape)
        entropy = np.asarray(fluid_model.computeEntropy_p_rho(p_flat, rho_flat)).reshape(shape)
        mach = np.asarray(fluid_model.computeMach_u_p_rho(u_flat, p_flat, rho_flat)).reshape(shape)

    state_by_key = {
        "Density": density,
        "Pressure": pressure,
        "Velocity": velocity,
        "staticInternalEnergy": energy,
        "Temperature": temperature,
        "Entropy": entropy,
        "Mach": mach,
    }
    profiles = state_by_key[_profile_key(fluid_state_var_name)]  # (nodes, frames)

    # customdata: (nodes, frames, 1 + n_fields), mesh node number first.
    fields = np.stack([state_by_key[key] for _, key, _ in _STATE_FIELDS], axis=-1)
    node_numbers = np.broadcast_to(
        mesh_node_numbers[:, None, None].astype(fields.dtype), (*shape, 1)
    )
    customdata = np.concatenate([node_numbers, fields], axis=-1)
    if compact_hover_data:
        customdata = customdata.astype(np.float32)

    masks = (
        _unconverged_masks(history, config, show_ghost_nodes)
        if show_unconverged_nodes
        else None
    )

    # --- Axis range from all finite values.
    finite_values = profiles[np.isfinite(profiles)]
    if finite_values.size == 0:
        raise ValueError(
            f"No finite values found for '{fluid_state_var_name}' in "
            f"'{results_folder}'."
        )
    value_min = float(finite_values.min())
    value_max = float(finite_values.max())
    value_span = value_max - value_min
    margin = 0.05 * value_span if value_span else max(abs(value_min) * 0.05, 1.0)
    variable_label, unit = _PROFILE_LABELS[fluid_state_var_name]
    hover_template = _hover_template(include_mesh_node=True)
    time_history = np.asarray(history["timeHistory"])

    def frame_marker(frame_index: int):
        if masks is None:
            return None
        return {"color": np.where(masks[:, frame_index], "red", "green")}

    def frame_title(frame_index: int) -> str:
        history_index = frame_indexes[frame_index]
        return (
            f"{variable_label} profile: "
            f"iteration={iteration_indexes[history_index]}, "
            f"time={time_history[history_index]:.6g} s"
        )

    # Frames carry only what changes; static trace properties live in the
    # initial trace (Plotly merges frame data into trace 0).
    frames = [
        go.Frame(
            name=str(frame_index),
            data=[go.Scatter(
                y=profiles[:, history_index],
                customdata=customdata[:, history_index, :],
                marker=frame_marker(history_index),
            )],
            traces=[0],
            layout={"title": frame_title(frame_index)},
        )
        for frame_index, history_index in enumerate(frame_indexes)
    ]

    initial_trace = go.Scatter(
        x=x_values,
        y=profiles[:, frame_indexes[0]],
        mode="lines+markers",
        name=variable_label,
        customdata=customdata[:, frame_indexes[0], :],
        hovertemplate=hover_template,
        marker=frame_marker(frame_indexes[0]),
    )

    figure = go.Figure(data=[initial_trace], frames=frames)
    figure.update_layout(
        template="plotly_white",
        xaxis_title="x [m]",
        yaxis_title=f"{variable_label} [{unit}]" if unit != "-" else variable_label,
        xaxis={"range": [float(x_values.min()), float(x_values.max())]},
        yaxis={"range": [value_min - margin, value_max + margin]},
        title=frame_title(0),
        hovermode="closest",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.1,
                "y": 1.15,
                "buttons": [
                    {
                        "label": f"Play {speed:g}x",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": max(int(100 / speed), 1),
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                            },
                        ],
                    }
                    for speed in (0.5, 1.0, 2.0)
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.1,
                "y": -0.08,
                "len": 0.8,
                "currentvalue": {"prefix": "Iteration: "},
                "steps": [
                    {
                        "label": str(iteration_indexes[history_index]),
                        "method": "animate",
                        "args": [
                            [str(frame_index)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for frame_index, history_index in enumerate(frame_indexes)
                ],
            }
        ],
    )
    return figure


# --------------------------------------------------------------------------
# TS expansion plot
# --------------------------------------------------------------------------

def generate_expansion_thermoplot_plotly(
    results_folders: list[str],
    iteration_indexes: list[int],
    legend_labels: list[str],
    thermoplot_config_file_path: str,
    show_ghost_nodes: bool = False,
) -> go.Figure:
    """Create an interactive Plotly TS expansion plot from existing thermoplot output."""
    histories = collect_results_by_folder(results_folders, iteration_indexes)
    results = extract_results_at_iteration_indexes(histories, iteration_indexes)
    if len(legend_labels) != len(results):
        raise ValueError("legend_labels must match results_folders")

    result_list = list(results.values())
    fluids = [result["config"].fluidName() for result in result_list]
    if len(set(fluids)) != 1:
        raise ValueError("All result folders must use the same working fluid")

    thermoplot_config = ConfigThermoplot(thermoplot_config_file_path)
    thermoplot_config.get_thermoplot_settings()
    if thermoplot_config.thermoplot_settings["diagram_type"] != "TS":
        raise ValueError("Expansion paths require a TS thermoplot configuration")
    if fluids[0] != thermoplot_config.thermoplot_settings["fluid_name"]:
        raise ValueError("Simulation and thermoplot fluids must match")

    # All results share the same fluid, so one fluid model serves them all.
    fluid_model = _fluid_model(result_list[0]["config"])
    states = [_state_values(result, show_ghost_nodes, fluid_model) for result in result_list]

    finite_entropy = [s["Entropy"][np.isfinite(s["Entropy"])] for s in states]
    finite_temperature = [s["Temperature"][np.isfinite(s["Temperature"])] for s in states]
    settings = {
        "S_range": [min(v.min() for v in finite_entropy) * 0.80,
                    max(v.max() for v in finite_entropy) * 1.20],
        "T_range": [min(v.min() for v in finite_temperature) * 0.80,
                    max(v.max() for v in finite_temperature) * 1.20],
    }

    matplotlib_figure = thermoplot_cached(
        thermoplot_config_file_path,
        thermoplot_overwrite_settings=settings,
    )
    figure = mpl_to_plotly(matplotlib_figure)

    # The horizontal axis of this plot is entropy, not position.
    hover_template = _hover_template().replace(
        "x: %{x:.6g} m", "s: %{x:.6g} J/(kg K)"
    )
    for label, state in zip(legend_labels, states):
        figure.add_trace(go.Scatter(
            x=state["Entropy"],
            y=state["Temperature"],
            mode="lines+markers",
            name=label,
            customdata=_hover_customdata(state),
            hovertemplate=hover_template,
        ))

    figure.update_layout(hovermode="closest")
    return figure