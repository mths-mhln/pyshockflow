###########################################
# Imports
###########################################
import copy

from configthermoplot import ConfigThermoplot
from general_helpers import extract_critical_point
from coolprop_interface_thermoplot import CoolPropAbstractState

import numpy as np





###########################################
# Isoline generation - General
###########################################
def construct_saturation_dome(config: type[ConfigThermoplot], AS: type[CoolPropAbstractState]) -> np.ndarray:
    # independent variable is the vertical axis variable in the diagram. This is necessary to obtain determinate thdy pairs. 
    iv_type = config.thermoplot_settings["diagram_type"][0]
    dv_type = config.thermoplot_settings["diagram_type"][-1]

    # extract independent variable range and get control points of iv at which to evaluate isoline.
    # dome extends only to critical point, so place that as upper bound.
    iv_range = copy.deepcopy(config.thermoplot_settings[f"{iv_type}_range"])
    iv_crit = AS.PropsSI(f"{iv_type}crit")
    iv_range[1] = min(iv_range[1], iv_crit)

    # Split range into segment until 5% below critical point. *2 for additional resolution
    if config.thermoplot_settings["diagram_type"] == "TS":
        iv_cp = np.linspace(iv_range[0], iv_range[1], config.thermoplot_settings["n_pts"]*2)
    elif config.thermoplot_settings["diagram_type"] == "PH":
        iv_cp = np.geomspace(iv_range[0], iv_range[1], config.thermoplot_settings["n_pts"]*2)

    # evaluate saturation dome branches left and right of critical point at control points. 
    dome_coords_Q_0 = np.column_stack((AS.PropsSI(dv_type, iv_type, iv_cp, "Q", 0), iv_cp)) # Q = 0 isoline
    dome_coords_Q_1 = np.column_stack((AS.PropsSI(dv_type, iv_type, iv_cp, "Q", 1), iv_cp)) # Q = 1 isoline

    # evaluate critical point coordinates and combine with dome branches
    crit_coords = extract_critical_point(config, AS)
    dome_coords = np.concatenate([dome_coords_Q_0, crit_coords, dome_coords_Q_1[::-1]]) # combine into single array of coordinates for saturation dome

    return dome_coords



def construct_quality_isolines(config: type[ConfigThermoplot], AS: type[CoolPropAbstractState], n_iq_lines: float) -> np.ndarray:
    # independent variable is the vertical axis variable in the diagram. This is necessary to obtain determinate thdy pairs. 
    iv_type = config.thermoplot_settings["diagram_type"][0]
    dv_type = config.thermoplot_settings["diagram_type"][-1]

    # extract independent variable range and get control points of iv at which to evaluate isoline.
    # dome extends only to critical point, so place that as upper bound.
    iv_range = copy.deepcopy(config.thermoplot_settings[f"{iv_type}_range"])
    iv_crit = AS.PropsSI(f"{iv_type}crit")
    iv_range[1] = min(iv_range[1], iv_crit)

    if config.thermoplot_settings["diagram_type"] == "TS":
        iv_cp = np.linspace(iv_range[0], iv_range[1], config.thermoplot_settings["n_pts"])
    elif config.thermoplot_settings["diagram_type"] == "PH":
        iv_cp = np.geomspace(iv_range[0], iv_range[1], config.thermoplot_settings["n_pts"])

    # instantiate quality isolines list
    quality_isolines_data = []

    # construct quality isolines. 
    for q in np.linspace(0, 1, n_iq_lines):
        # evaluate isoline at control points. 
        quality_arr = np.ones(iv_cp.shape) * q
        dv_vals = AS.PropsSI(dv_type, iv_type, iv_cp, "Q", quality_arr)

        # combine iv and dv into coordinate pairs and return
        quality_isolines_coords = np.column_stack((dv_vals, iv_cp))

        # store quality value and isoline in dictionary
        quality_isolines_data.append({"isoline_val": q, "coords": quality_isolines_coords})

    return quality_isolines_data



def construct_critical_isoline(config: type[ConfigThermoplot], AS: type[CoolPropAbstractState], n_pts: int) -> np.ndarray:
    # independent variable is the horizontal axis variable in the diagram. this is necessary as critical isoline reaches 0 slope
    iv_type = config.thermoplot_settings["diagram_type"][-1]
    dv_type = config.thermoplot_settings["diagram_type"][0]

    # Extract plot bounds
    iv_lo, iv_hi = config.thermoplot_settings[f"{iv_type}_range"]
    dv_lo, dv_hi = config.thermoplot_settings[f"{dv_type}_range"]

    # get critical isoline type
    if config.thermoplot_settings["diagram_type"] == "TS":
        critical_isoline_type = "P"
    elif config.thermoplot_settings["diagram_type"] == "PH":
        critical_isoline_type = "T"

    # extract independent variable range and get control points of iv at which to evaluate isoline.
    # dome extends only to critical point, so place that as upper bound.
    iv_range = copy.deepcopy(config.thermoplot_settings[f"{iv_type}_range"])
    iv_cp = np.linspace(iv_range[0], iv_range[1], n_pts)

    # get critical isoline value
    critical_isoline_val = AS.PropsSI(f"{critical_isoline_type}crit")

    # evaluate dv at control points for critical quality of 0.5
    critical_isoline_arr = np.ones(iv_cp.shape) * critical_isoline_val
    dv_vals = AS.PropsSI(dv_type, iv_type, iv_cp, critical_isoline_type, critical_isoline_arr)

    # only keep vals inside of plot range
    v = (np.isfinite(dv_vals) & (dv_vals >= dv_lo) & (dv_vals <= dv_hi) &
         (iv_cp >= iv_lo*0.99) & (iv_cp <= iv_hi*1.01))

    # combine iv and dv into coordinate pairs and return
    critical_isoline_coords = np.column_stack((iv_cp[v], dv_vals[v]))

    return critical_isoline_coords





###########################################
# Isoline generation - TS
###########################################
def isobar_lines_ts(config: type[ConfigThermoplot], AS: type[CoolPropAbstractState], n_lines=12):
    # extract plot limits
    T_lo, T_hi = config.thermoplot_settings["T_range"]
    s_lo, s_hi = config.thermoplot_settings["S_range"]

    # compute 
    p_crit = AS.PropsSI("Pcrit")
    T_trip = AS.PropsSI("Ttriple")
    p_trip = AS.PropsSI("P", "T", T_trip*1.001, "Q", 0)
    p_lo_est = max(p_trip*1.1 if np.isfinite(p_trip) else 1e3, 1e3)
    p_vals = np.geomspace(p_lo_est, p_crit*3.0, n_lines)

    T_sweep = np.linspace(T_lo,T_hi, config.thermoplot_settings["n_pts"])
    
    #instantiate
    isobar_lines_data_ts = []
    for p in p_vals:
        if p < p_crit:
            T_sat = AS.PropsSI("T", "P", p, "Q", 0)
            s_sat_l = AS.PropsSI("S", "P", p, "Q", 0)
            s_sat_v = AS.PropsSI("S", "P", p, "Q", 1)
            if not (np.isfinite(T_sat) and np.isfinite(s_sat_l) and np.isfinite(s_sat_v)):
                continue
            # Subcooled branch (T < T_sat)
            T_liq = T_sweep[T_sweep < T_sat]
            s_liq = np.array([AS.PropsSI("S", "P", p, "T", T) for T in T_liq])
            # Superheated branch (T > T_sat)
            T_vap = T_sweep[T_sweep > T_sat]
            s_vap = np.array([AS.PropsSI("S", "P", p, "T", T) for T in T_vap])
            # Two-phase horizontal bridge at T_sat
            s_2ph = np.linspace(s_sat_l, s_sat_v, 20)
            T_2ph = np.full(20, T_sat)
            # Stitch in physical order to avoid artificial loops/crossings.
            s_all = np.concatenate([s_liq, s_2ph, s_vap])
            T_all = np.concatenate([T_liq, T_2ph, T_vap])
        else:
            s_all = np.array([AS.PropsSI("S", "P", p, "T", T) for T in T_sweep])
            T_all = T_sweep

        v = (np.isfinite(s_all) & (s_all >= s_lo) & (s_all <= s_hi) &
             (T_all >= T_lo*0.95) & (T_all <= T_hi*1.05))
        if v.sum() > 3:
            isobar_isoline_coords = np.column_stack((s_all[v], T_all[v]))
            isobar_lines_data_ts.append({"isoline_val": p,"coords": isobar_isoline_coords})
    return isobar_lines_data_ts


def isenthalp_lines_ts(config: type[ConfigThermoplot], AS: type[CoolPropAbstractState], n_lines=18):
    # extract plot limits
    T_lo, T_hi = config.thermoplot_settings["T_range"]
    s_lo, s_hi = config.thermoplot_settings["S_range"]

    # Sample h range from corners of the visible TS window
    h_samples = []
    for s in np.linspace(s_lo, s_hi, 6):
        for T in np.linspace(T_lo, T_hi, 6):
            try:
                h_samples.append(AS.PropsSI("H", "T", T, "S", s))
            except:
                h_samples.append(np.nan)
    h_samples = [h for h in h_samples if np.isfinite(h)]
    if not h_samples:
        return []
    h_vals  = np.linspace(min(h_samples)*0.95, max(h_samples)*1.05, n_lines)
    p_crit  = AS.PropsSI("Pcrit")
    p_arr   = np.geomspace(p_crit*3.0, 1e3, config.thermoplot_settings["n_pts"])

    # instantiate
    isenthalp_lines_data_ts = []

    for h in h_vals:
        s_arr = np.array([AS.PropsSI("S", "P", p, "H", h) for p in p_arr])
        T_arr = np.array([AS.PropsSI("T", "P", p, "H", h) for p in p_arr])
        v = (np.isfinite(s_arr) & np.isfinite(T_arr) &
             (s_arr >= s_lo) & (s_arr <= s_hi) &
             (T_arr >= T_lo*0.99) & (T_arr <= T_hi*1.01))
        if v.sum() > 3:
            isenthalp_isoline_coords = np.column_stack((s_arr[v], T_arr[v]))
            isenthalp_lines_data_ts.append({"isoline_val": h,"coords": isenthalp_isoline_coords})

    return isenthalp_lines_data_ts





###########################################
# Isoline generation - PH
###########################################
def isotherm_lines_ph(config: type[ConfigThermoplot], AS: type[CoolPropAbstractState], n_lines=18):
    # extract plot limits
    p_lo, p_hi = config.thermoplot_settings["P_range"]
    h_lo, h_hi = config.thermoplot_settings["H_range"]

    # evaluate
    T_crit = AS.PropsSI("Tcrit")
    T_trip = AS.PropsSI("Ttriple", "T", 300, "Q", 1)
    T_vals = np.linspace(T_trip*1.05, T_crit*1.5, n_lines)
    p_arr  = np.geomspace(max(p_lo*0.5, 1e3), p_hi*1.5, config.thermoplot_settings["n_pts"])

    # instantiate
    isotherm_lines_data_ph = []
    for T in T_vals:
        if T < T_crit:
            p_sat = AS.PropsSI("P", "T", T, "Q", 0)
            h_sat_l = AS.PropsSI("H", "T", T, "Q", 0)
            h_sat_v = AS.PropsSI("H", "T", T, "Q", 1)
            if not (np.isfinite(p_sat) and np.isfinite(h_sat_l) and np.isfinite(h_sat_v)):
                continue
            # Liquid branch (p > p_sat)
            p_liq = p_arr[p_arr > p_sat]
            h_liq = np.array([AS.PropsSI("H", "P", p, "T", T) for p in p_liq])
            # Vapour branch (p < p_sat)
            p_vap = p_arr[p_arr < p_sat]
            h_vap = np.array([AS.PropsSI("H", "P", p, "T", T) for p in p_vap])
            # Two-phase horizontal bridge at p_sat
            h_2ph = np.linspace(h_sat_l, h_sat_v, 20)
            p_2ph = np.full(20, p_sat)
            # Stitch: liquid (high→p_sat) → bridge → vapour (p_sat→low)
            h_all = np.concatenate([h_liq[::-1], h_2ph, h_vap[::-1]])
            p_all = np.concatenate([p_liq[::-1], p_2ph, p_vap[::-1]])
        else:
            # flip p_arr for correct label orientation
            p_arr = p_arr[::-1]
            h_all = np.array([AS.PropsSI("H", "P", p, "T", T) for p in p_arr])
            p_all = p_arr

        v = (np.isfinite(h_all) &
             (h_all >= h_lo) & (h_all <= h_hi) &
             (p_all >= p_lo*0.9) & (p_all <= p_hi*1.1))
        if v.sum() > 3:
            isotherm_isoline_coords = np.column_stack((h_all[v], p_all[v]))
            isotherm_lines_data_ph.append({"isoline_val": T,"coords": isotherm_isoline_coords})
    return isotherm_lines_data_ph


def isentrop_lines_ph(config: type[ConfigThermoplot], AS: type[CoolPropAbstractState], n_lines=12):
    # extract plot limits
    p_lo, p_hi = config.thermoplot_settings["P_range"]
    h_lo, h_hi = config.thermoplot_settings["H_range"]

    s_samples = []
    for h in np.linspace(h_lo, h_hi, 6):
        for p in np.geomspace(max(p_lo, 1e3), p_hi, 6):
            s_samples.append(AS.PropsSI("S", "P", p, "H", h))
    s_samples = [s for s in s_samples if np.isfinite(s)]
    if not s_samples:
        return []
    s_vals = np.linspace(min(s_samples), max(s_samples), n_lines)
    p_arr  = np.geomspace(max(p_lo*0.5, 1e3), p_hi*1.5, config.thermoplot_settings["n_pts"])
    isentrop_lines_data_ph = []
    for sv in s_vals:
        h_arr = np.array([AS.PropsSI("H", "P", p, "S", sv) for p in p_arr])
        v = (np.isfinite(h_arr) &
             (h_arr >= h_lo) & (h_arr <= h_hi) &
             (p_arr >= p_lo*0.9) & (p_arr <= p_hi*1.1))
        if v.sum() > 3:
            isentrop_isoline_coords = np.column_stack((h_arr[v], p_arr[v]))
            isentrop_lines_data_ph.append({"isoline_val": sv,"coords": isentrop_isoline_coords})
    return isentrop_lines_data_ph