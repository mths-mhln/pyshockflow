"""Interactive Plotly visualizations for PyShockFlow results.

The existing Matplotlib plotting functions remain unchanged.  These helpers
reuse their result extraction and fluid-property calculations while adding
complete SI-unit state information to Plotly hover labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
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
    "internalEnergy": ("Internal energy", "J/kg"),
    "Temperature": ("Temperature", "K"),
    "Entropy": ("Entropy", "J/(kg K)"),
    "Mach": ("Mach", "-"),
}


def _state_values(results_at_iteration: dict, show_ghost_nodes: bool) -> dict[str, np.ndarray]:
    """Return native and derived state fields on the requested mesh nodes."""
    node_slice = slice(None) if show_ghost_nodes else slice(1, -1)
    fluid_state = results_at_iteration["fluidState"]

    with np.errstate(invalid="ignore", divide="ignore"):
        with _hidden_prints():
            driver = Driver(config=results_at_iteration["config"])
        pressure = np.asarray(fluid_state["Pressure"])[node_slice]
        density = np.asarray(fluid_state["Density"])[node_slice]
        velocity = np.asarray(fluid_state["Velocity"])[node_slice]

        return {
            "Density": density,
            "Pressure": pressure,
            "Velocity": velocity,
            "staticInternalEnergy": np.asarray(fluid_state["staticInternalEnergy"])[node_slice],
            "Temperature": np.asarray(driver.fluidModel.computeTemperature_p_rho(pressure, density)),
            "Entropy": np.asarray(driver.fluidModel.computeEntropy_p_rho(pressure, density)),
            "Mach": np.asarray(driver.fluidModel.computeMach_u_p_rho(velocity, pressure, density)),
        }


class _hidden_prints:
    """Suppress driver construction output without changing post_processing."""

    def __enter__(self):
        import contextlib
        import io

        self._redirect = contextlib.redirect_stdout(io.StringIO())
        self._redirect.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._redirect.__exit__(exc_type, exc_value, traceback)


def _hover_customdata(
    state: dict[str, np.ndarray],
    mesh_node_numbers: Optional[np.ndarray] = None,
) -> np.ndarray:
    columns = [state[key] for _, key, _ in _STATE_FIELDS]
    if mesh_node_numbers is not None:
        columns.insert(0, mesh_node_numbers)
    return np.column_stack(columns)


def _hover_template(include_mesh_node: bool = False) -> str:
    lines = [
        "<b>%{fullData.name}</b>",
    ]
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


def generate_fluid_state_var_profile_plotly(
    results_folders: list[str],
    fluid_state_var_names: list[str],
    iteration_indexes: list[int],
    simulation_legend_labels: Optional[list[str]] = None,
    verification_data: Optional[VandVSpec] = None,
    validation_data: Optional[VandVSpec] = None,
    show_nozzle_geometry: bool = False,
    show_ghost_nodes: bool = False,
) -> list[go.Figure]:
    """Create interactive spatial profiles with complete state hover labels.

    One Plotly figure is returned for each requested profile.  Hovering a
    simulation point shows density, pressure, velocity, internal energy,
    temperature, entropy, and Mach number, all in SI units. Optional
    verification and validation records use the same `VandVSpec` and CSV
    parsing as the Matplotlib profile plots.
    """
    unknown = set(fluid_state_var_names) - set(_PROFILE_LABELS)
    if unknown:
        raise ValueError(f"Unsupported fluid state variables: {sorted(unknown)}")

    histories = collect_results_by_folder(results_folders, iteration_indexes)
    results = extract_results_at_iteration_indexes(histories, iteration_indexes)
    if simulation_legend_labels is None:
        simulation_legend_labels = list(results)
    if len(simulation_legend_labels) != len(results):
        raise ValueError("simulation_legend_labels must match results_folders")

    figures = []
    hover_template = _hover_template(include_mesh_node=True)
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
        figure = go.Figure()
        last_result = None
        max_profile = 0.0

        for label, result in zip(simulation_legend_labels, results.values()):
            state = _state_values(result, show_ghost_nodes)
            node_slice = slice(None) if show_ghost_nodes else slice(1, -1)
            mesh_nodes = np.asarray(result["meshData"]["xMeshNodes"])
            x_values = mesh_nodes[node_slice]
            mesh_node_numbers = np.arange(mesh_nodes.size)[node_slice]
            key = "staticInternalEnergy" if variable == "internalEnergy" else variable
            profile = state[key]
            max_profile = max(max_profile, float(np.nanmax(np.abs(profile))))
            figure.add_trace(go.Scatter(
                x=x_values,
                y=profile,
                mode="lines+markers",
                name=f"{label}: iteration={result['iterIdx']}",
                customdata=_hover_customdata(state, mesh_node_numbers),
                hovertemplate=hover_template,
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
            node_slice = slice(None) if show_ghost_nodes else slice(1, -1)
            area = np.asarray(last_result["meshData"]["deviceAreaAtMeshNodes"])[node_slice]
            figure.add_trace(go.Scatter(
                x=np.asarray(last_result["meshData"]["xMeshNodes"])[node_slice],
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


def generate_fluid_state_var_profile_animation_plotly(
    results_folder: str,
    fluid_state_var_name: str,
    show_ghost_nodes: bool = False,
    interval: int = 100,
) -> go.Figure:
    """Create a browser-based animated fluid-state profile.

    The returned Plotly figure includes an iteration slider, play controls,
    and playback buttons for half, normal, and double speed. Hover labels
    include the mesh-node number and all available state variables.
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

    mesh_nodes = np.asarray(history["meshData"]["xMeshNodes"])
    node_slice = slice(None) if show_ghost_nodes else slice(1, -1)
    x_values = mesh_nodes[node_slice]
    mesh_node_numbers = np.arange(mesh_nodes.size)[node_slice]
    profiles = []
    states = []

    for frame_index in range(iteration_indexes.size):
        fluid_state = {
            key: values[:, frame_index]
            for key, values in history["fluidStateHistory"].items()
        }
        snapshot = {
            "config": history["config"],
            "fluidState": fluid_state,
        }
        state = _state_values(snapshot, show_ghost_nodes)
        key = (
            "staticInternalEnergy"
            if fluid_state_var_name == "internalEnergy"
            else fluid_state_var_name
        )
        profiles.append(np.asarray(state[key]))
        states.append(state)

    finite_values = np.concatenate(
        [profile[np.isfinite(profile)] for profile in profiles]
    )
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

    def trace_for(frame_index: int) -> go.Scatter:
        return go.Scatter(
            x=x_values,
            y=profiles[frame_index],
            mode="lines+markers",
            name=variable_label,
            customdata=_hover_customdata(states[frame_index], mesh_node_numbers),
            hovertemplate=hover_template,
        )

    time_history = np.asarray(history["timeHistory"])
    frames = [
        go.Frame(
            name=str(frame_index),
            data=[trace_for(frame_index)],
            layout={
                "title": (
                    f"{variable_label} profile: "
                    f"iteration={iteration_indexes[frame_index]}, "
                    f"time={time_history[frame_index]:.6g} s"
                )
            },
        )
        for frame_index in range(iteration_indexes.size)
    ]

    figure = go.Figure(data=[trace_for(0)], frames=frames)
    figure.update_layout(
        template="plotly_white",
        xaxis_title="x [m]",
        yaxis_title=f"{variable_label} [{unit}]" if unit != "-" else variable_label,
        xaxis={"range": [float(x_values.min()), float(x_values.max())]},
        yaxis={"range": [value_min - margin, value_max + margin]},
        title=(
            f"{variable_label} profile: iteration={iteration_indexes[0]}, "
            f"time={time_history[0]:.6g} s"
        ),
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
                                    "duration": max(int(interval / speed), 1),
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
                        "label": str(iteration_index),
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
                    for frame_index, iteration_index in enumerate(iteration_indexes)
                ],
            }
        ],
    )
    return figure


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

    fluids = [result["config"].fluidName() for result in results.values()]
    if len(set(fluids)) != 1:
        raise ValueError("All result folders must use the same working fluid")

    thermoplot_config = ConfigThermoplot(thermoplot_config_file_path)
    thermoplot_config.get_thermoplot_settings()
    if thermoplot_config.thermoplot_settings["diagram_type"] != "TS":
        raise ValueError("Expansion paths require a TS thermoplot configuration")
    if fluids[0] != thermoplot_config.thermoplot_settings["fluid_name"]:
        raise ValueError("Simulation and thermoplot fluids must match")

    states = []
    for result in results.values():
        states.append(_state_values(result, show_ghost_nodes))

    entropy_values = [state["Entropy"] for state in states]
    temperature_values = [state["Temperature"] for state in states]
    finite_entropy = [values[np.isfinite(values)] for values in entropy_values]
    finite_temperature = [values[np.isfinite(values)] for values in temperature_values]
    settings = {
        "S_range": [min(values.min() for values in finite_entropy) * 0.80,
                    max(values.max() for values in finite_entropy) * 1.20],
        "T_range": [min(values.min() for values in finite_temperature) * 0.80,
                    max(values.max() for values in finite_temperature) * 1.20],
    }

    matplotlib_figure = thermoplot_cached(
        thermoplot_config_file_path,
        thermoplot_overwrite_settings=settings,
    )
    figure = mpl_to_plotly(matplotlib_figure)
    hover_template = _hover_template().replace("x =", "s =")
    for label, state in zip(legend_labels, states):
        figure.add_trace(go.Scatter(
            x=state["Entropy"],
            y=state["Temperature"],
            mode="lines+markers",
            name=label,
            customdata=_hover_customdata(state),
            hovertemplate=hover_template.replace(" m<", " J/(kg K)<"),
        ))

    figure.update_layout(hovermode="closest")
    return figure