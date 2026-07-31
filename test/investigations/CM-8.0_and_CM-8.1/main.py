# necessary components: 
# 1) fluid property extraction method -> look at pytest how to properly initialize FP object. 
# 2) thermoplot saturation dome calculation scheme
# 2.5) compute critical point and split saturation dome into LHS and RHS of crit point.
# 2.75) interpolate the LHS and RHS of the saturation dome to get S min and max values per T linspace value.
# 3) linspace of T between T_min and T_max of saturation dome
# 4) per T, linspace-like S values at which to evaluate rho, u. 
# 5) generation of dictionary saving coordinate on T-S (the T, S pair) and associated rho, u values using Propssi.
# 6) Computation of entropy values using propsSi, and sanity check that these are the same S as the coordinate S values
# 7) Computation of entropy using Giljarhus system of equations, saving to dictionary
# 8) heat map generation showing the rel deviation from Propssi and Giljarhus system of equations.
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import brentq

from fluid_properties.coolprop_interface import CoolPropAbstractState_v2
from thermoplot.isolines import construct_saturation_dome
from thermoplot.thermoplot import thermoplot_cached
from thermoplot.configthermoplot import ConfigThermoplot



# instantiate abstractstate object, allows for calculation of 
# thdy properties using the PropsSI method
AS = CoolPropAbstractState_v2("REFPROP", "R1234ze(E)")

# Instantiate thermoplot config object according to user settings
# Extract saturation dome using thermoplot functionality
input_file_path = "config/R1234ze(E).ini"
config = ConfigThermoplot(config_file=input_file_path)
config.get_thermoplot_settings()
dome_coords = construct_saturation_dome(config, AS)

# compute critical point and split saturation dome into LHS and 
# RHS of crit point. extract the (S, T) coordinates
T_crit = AS.PropsSI("Tcrit")
crit_idx = np.argmin(np.abs(dome_coords[:, 1] - T_crit))
saturation_dome_LHS = dome_coords[:crit_idx + 1, :]
saturation_dome_RHS = dome_coords[crit_idx:, :]

# interpolate saturation dome LHS and RHS coordinates to extract
# saturation dome S ordinate at T ordinate of interest.
f_LHS = interp1d(saturation_dome_LHS[:, 1], saturation_dome_LHS[:, 0], kind='linear', fill_value='extrapolate')
f_RHS = interp1d(saturation_dome_RHS[:, 1], saturation_dome_RHS[:, 0], kind='linear', fill_value='extrapolate')

# initiate dictionary in which to store the investigation data
investigation_data = {
    "ST_coords": None,
    "rho_values": None,
    "u_values": None,
    "S_propssi": None,
    "S_giljarhus": None,
    "P_propssi": None,
    "P_giljarhus": None,
    "U_propssi": None,
    "U_giljarhus": None,
    "alpha_propssi": None,
    "alpha_giljarhus": None,
    "rho_giljarhus": None,
    "rho_propssi": None
}

# find lowest / highest T achievable along the dome. Used as the outer
# fallback bracket for the 1D root find below (triple-point-ish to Tcrit).
T_dome_min = min(dome_coords[:, 1])
T_dome_max = max(dome_coords[:, 1])

T_min = T_dome_min
T_max = T_dome_max
T_values = np.linspace(T_min, T_max, 500)

# find largest S diff present in the saturation dome:
S_diffs = []
for T in T_values:
    S_min = f_LHS(T)
    S_max = f_RHS(T)
    S_diffs.append(S_max - S_min)
max_S_diff = max(S_diffs)


# ---------------------------------------------------------------------------
# Analytic elimination of alpha + rho_V/rho_L from the Giljarhus system.
#
# For a pure fluid, eq3 (P_L = P_V) and eq4 (G_L = G_V) simply say that
# (rho_V, rho_L) sit on the saturation curve at temperature T, i.e.
#   rho_V(T) = PropsSI("D", "T", T, "Q", 1)
#   rho_L(T) = PropsSI("D", "T", T, "Q", 0)
# These are not free unknowns. Eq2 is linear in alpha, so it can be solved
# for alpha(T) in closed form. Substituting into eq1 leaves a single
# scalar equation in the single unknown T, which is well-conditioned and
# solved with brentq (bounded, guaranteed convergence given a valid
# bracket -- no Jacobian/scaling issues, no wandering into nonphysical
# densities).
# ---------------------------------------------------------------------------

def _alpha_of_T(T, rho_in):
    """Vapor volume fraction implied by saturation densities at T."""
    rho_V = AS.PropsSI("D", "T", T, "Q", 1, verbose=False)
    rho_L = AS.PropsSI("D", "T", T, "Q", 0, verbose=False)
    alpha = (rho_in - rho_L) / (rho_V - rho_L)
    return alpha, rho_V, rho_L


def _residual(T, rho_in, u_in):
    """Energy-balance residual (eq1) as a function of T alone."""
    alpha, rho_V, rho_L = _alpha_of_T(T, rho_in)
    U_V = AS.PropsSI("U", "T", T, "Q", 1, verbose=False)
    U_L = AS.PropsSI("U", "T", T, "Q", 0, verbose=False)
    return alpha * rho_V * U_V + (1 - alpha) * rho_L * U_L - rho_in * u_in


def _find_bracket(rho_in, u_in, T_center, T_lo, T_hi, n_probe=25):
    """
    Find a [T_a, T_b] pair with a sign change in the residual, searching
    outward from T_center first (cheap, usually succeeds immediately since
    the equilibrium T is close to the nominal iso-T of the current sweep),
    then falling back to a scan of the full dome range.
    """
    # cheap local attempt: symmetric expansion around T_center
    for half_width in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        Ta = max(T_lo, T_center - half_width)
        Tb = min(T_hi, T_center + half_width)
        if Ta == Tb:
            continue
        try:
            fa = _residual(Ta, rho_in, u_in)
            fb = _residual(Tb, rho_in, u_in)
        except ValueError:
            continue
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            return Ta, Tb

    # fallback: scan the full dome range for a sign change
    T_grid = np.linspace(T_lo, T_hi, n_probe)
    f_grid = []
    for T in T_grid:
        try:
            f_grid.append(_residual(T, rho_in, u_in))
        except ValueError:
            f_grid.append(np.nan)
    f_grid = np.array(f_grid)
    for i in range(len(T_grid) - 1):
        fa, fb = f_grid[i], f_grid[i + 1]
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            return T_grid[i], T_grid[i + 1]

    return None  # no bracket found -- point likely too close to crit point


def solve_giljarhus_1D(rho_in, u_in, T_center, T_lo, T_hi):
    """
    Solve the Giljarhus system for a single (rho_in, u_in) pair via 1D
    root-finding in T. Returns (T_sol, rho_V_sol, rho_L_sol, alpha_sol),
    or None if no valid bracket could be found (e.g. too close to Tcrit).
    """
    bracket = _find_bracket(rho_in, u_in, T_center, T_lo, T_hi)
    if bracket is None:
        return None
    Ta, Tb = bracket
    T_sol = brentq(_residual, Ta, Tb, args=(rho_in, u_in), xtol=1e-8, maxiter=200)
    alpha_sol, rho_V_sol, rho_L_sol = _alpha_of_T(T_sol, rho_in)
    return T_sol, rho_V_sol, rho_L_sol, alpha_sol


def solve_giljarhus_system(rho_in_array, u_in_array, T_center, T_lo, T_hi):
    """Vectorized (elementwise) wrapper around solve_giljarhus_1D."""
    T_sol = np.empty_like(rho_in_array, dtype=float)
    rho_V_sol = np.empty_like(rho_in_array, dtype=float)
    rho_L_sol = np.empty_like(rho_in_array, dtype=float)
    alpha_sol = np.empty_like(rho_in_array, dtype=float)
    valid = np.ones(rho_in_array.shape, dtype=bool)

    for i in range(rho_in_array.size):
        result = solve_giljarhus_1D(rho_in_array[i], u_in_array[i], T_center, T_lo, T_hi)
        if result is None:
            valid[i] = False
            T_sol[i] = np.nan
            rho_V_sol[i] = np.nan
            rho_L_sol[i] = np.nan
            alpha_sol[i] = np.nan
        else:
            T_sol[i], rho_V_sol[i], rho_L_sol[i], alpha_sol[i] = result

    return T_sol, rho_V_sol, rho_L_sol, alpha_sol, valid


# at each of the temperature values:
for i, T in enumerate(T_values):
    # print evaluation status: chosen to be the T value under current evaluation.
    print(f"Processing T value {i+1}/{len(T_values)}: T = {T:.2f} K")

    # extract the S min and max values at the current T value, and compute the S diff.
    S_min = f_LHS(T)
    S_max = f_RHS(T)
    S_diff = S_max - S_min

    # extract the S values at which to evaluate the mixture properties.
    S_values = np.linspace(S_min, S_max, int(50 * S_diff / max_S_diff))
    if S_values.size == 0:  # this can happen near the critical point.
        continue

    # create an array of T values corresponding to S_values
    T_array = np.full_like(S_values, T)

    # combine S and T into a single array of coordinates
    coords = np.column_stack((S_values, T_array))

    # rho, u values corresponding to (T, S) via PropsSI -- treated as the
    # "input" mixture state that the Giljarhus system must reproduce.
    rho_values = np.array([AS.PropsSI("D", "T", T, "S", S, verbose=False) for S in S_values])
    u_values = np.array([AS.PropsSI("U", "T", T, "S", S, verbose=False) for S in S_values])

    # sanity check
    S_propssi = np.array([AS.PropsSI("S", "U", u, "D", rho, verbose=False) for u, rho in zip(u_values, rho_values)])
    # nan indexes
    nan_indexes = np.isnan(S_propssi)
    # set nan indexes to 0 on both
    S_propssi[nan_indexes] = 0
    S_values[nan_indexes] = 0
    assert np.allclose(S_propssi, S_values), f"Mismatch between computed S and coordinate S values using PropsSI. {S_propssi} vs {S_values}"

    # --- solve the Giljarhus system via 1D root-find in T ---
    T_sol, rho_V_sol, rho_L_sol, alpha_sol, valid = solve_giljarhus_system(
        rho_values, u_values, T_center=T, T_lo=T_dome_min, T_hi=T_dome_max
    )

    if not np.all(valid):
        n_bad = np.sum(~valid)
        print(f"  Warning: {n_bad} point(s) at T={T:.2f} K had no valid bracket "
              f"(likely too close to the critical point) -- skipping them.")
        S_values = S_values[valid]
        coords = coords[valid]
        rho_values = rho_values[valid]
        u_values = u_values[valid]
        S_propssi = S_propssi[valid]
        T_sol = T_sol[valid]
        rho_V_sol = rho_V_sol[valid]
        rho_L_sol = rho_L_sol[valid]
        alpha_sol = alpha_sol[valid]
        if S_values.size == 0:
            continue

    # sanity check that the solutions solve the governing equations
    resid_energy = np.abs((alpha_sol * rho_V_sol * AS.PropsSI("U", "T", T_sol, "D", rho_V_sol) +
                            (1 - alpha_sol) * rho_L_sol * AS.PropsSI("U", "T", T_sol, "D", rho_L_sol))
                           - rho_values * u_values) / (rho_values * u_values)
    resid_mass = np.abs(((alpha_sol * rho_V_sol + (1 - alpha_sol) * rho_L_sol) - rho_values) / rho_values)
    resid_P = np.abs((AS.PropsSI("P", "T", T_sol, "D", rho_V_sol) - AS.PropsSI("P", "T", T_sol, "D", rho_L_sol))
                      / AS.PropsSI("P", "T", T_sol, "D", rho_L_sol))
    resid_G = np.abs((AS.PropsSI("G", "T", T_sol, "D", rho_V_sol) - AS.PropsSI("G", "T", T_sol, "D", rho_L_sol))
                      / AS.PropsSI("G", "T", T_sol, "D", rho_L_sol))

    if (np.max(resid_energy) > 1e-2 or np.max(resid_mass) > 1e-2 or
            np.max(resid_P) > 1e-2 or np.max(resid_G) > 1e-2):
        raise ValueError("Solution does not satisfy the governing equations.")

    # calculate properties of interest to observe deviation

    # entropy
    S_V = AS.PropsSI("S", "T", T_sol, "Q", 1)
    S_L = AS.PropsSI("S", "T", T_sol, "Q", 0)
    Q = alpha_sol * AS.PropsSI("D", "T", T_sol, "Q", 1) / rho_values
    S_giljarhus = Q * S_V + (1 - Q) * S_L

    # pressure
    P_V = AS.PropsSI("P", "T", T_sol, "Q", 1)
    P_L = AS.PropsSI("P", "T", T_sol, "Q", 0)
    P_giljarhus = alpha_sol * P_V + (1 - alpha_sol) * P_L
    P_propssi = np.array([AS.PropsSI("P", "U", u, "D", rho) for u, rho in zip(u_values, rho_values)])

    # density
    rho_V = AS.PropsSI("D", "T", T_sol, "Q", 1)
    rho_L = AS.PropsSI("D", "T", T_sol, "Q", 0)
    rho_giljarhus = alpha_sol * rho_V + (1 - alpha_sol) * rho_L
    rho_propssi = rho_values

    # static internal energy
    U_V = AS.PropsSI("U", "T", T_sol, "Q", 1)
    U_L = AS.PropsSI("U", "T", T_sol, "Q", 0)
    U_giljarhus = Q * U_V + (1 - Q) * U_L
    U_propssi = np.array([AS.PropsSI("U", "P", p, "D", rho) for p, rho in zip(P_propssi, rho_values)])

    # volume fraction
    alpha_giljarhus = alpha_sol
    Q_propssi = np.array([AS.PropsSI("Q", "U", u, "D", rho) for u, rho in zip(u_values, rho_values)])
    alpha_propssi = Q_propssi * rho_values / AS.PropsSI("D", "T", T_sol, "Q", 1)

    # save to dictionary
    investigation_data["ST_coords"] = coords if investigation_data["ST_coords"] is None else np.concatenate((investigation_data["ST_coords"], coords), axis=0)
    investigation_data["rho_values"] = rho_values if investigation_data["rho_values"] is None else np.concatenate((investigation_data["rho_values"], rho_values), axis=0)
    investigation_data["u_values"] = u_values if investigation_data["u_values"] is None else np.concatenate((investigation_data["u_values"], u_values), axis=0)
    investigation_data["S_propssi"] = S_propssi if investigation_data["S_propssi"] is None else np.concatenate((investigation_data["S_propssi"], S_propssi), axis=0)
    investigation_data["S_giljarhus"] = S_giljarhus if investigation_data["S_giljarhus"] is None else np.concatenate((investigation_data["S_giljarhus"], S_giljarhus), axis=0)
    investigation_data["P_propssi"] = P_propssi if investigation_data["P_propssi"] is None else np.concatenate((investigation_data["P_propssi"], P_propssi), axis=0)
    investigation_data["P_giljarhus"] = P_giljarhus if investigation_data["P_giljarhus"] is None else np.concatenate((investigation_data["P_giljarhus"], P_giljarhus), axis=0)
    investigation_data["U_propssi"] = U_propssi if investigation_data["U_propssi"] is None else np.concatenate((investigation_data["U_propssi"], U_propssi), axis=0)
    investigation_data["U_giljarhus"] = U_giljarhus if investigation_data["U_giljarhus"] is None else np.concatenate((investigation_data["U_giljarhus"], U_giljarhus), axis=0)
    investigation_data["alpha_propssi"] = alpha_propssi if investigation_data["alpha_propssi"] is None else np.concatenate((investigation_data["alpha_propssi"], alpha_propssi), axis=0)
    investigation_data["alpha_giljarhus"] = alpha_giljarhus if investigation_data["alpha_giljarhus"] is None else np.concatenate((investigation_data["alpha_giljarhus"], alpha_giljarhus), axis=0)
    investigation_data["rho_giljarhus"] = rho_giljarhus if investigation_data["rho_giljarhus"] is None else np.concatenate((investigation_data["rho_giljarhus"], rho_giljarhus), axis=0)
    investigation_data["rho_propssi"] = rho_propssi if investigation_data["rho_propssi"] is None else np.concatenate((investigation_data["rho_propssi"], rho_propssi), axis=0)

print(investigation_data)

properties = ["rho", "P", "U", "alpha"]
plot_values = False

for property in properties:
    # generate boolean masks:
    mask_1 = np.abs(investigation_data[f"{property}_propssi"]) < np.full_like(investigation_data[f"{property}_propssi"], 1e-8)
    mask_2 = np.abs(investigation_data[f"{property}_giljarhus"]) < np.full_like(investigation_data[f"{property}_giljarhus"], 1e-8)
    mask_3 = np.abs(investigation_data[f"{property}_propssi"]) >= np.full_like(investigation_data[f"{property}_propssi"], 1e-8)
    mask_4 = np.abs(investigation_data[f"{property}_giljarhus"]) >= np.full_like(investigation_data[f"{property}_giljarhus"], 1e-8)

    # generate heat map showing the relative deviation from PropsSI and Giljarhus system of equations.
    deviation = np.zeros_like(investigation_data[f"{property}_propssi"])
    deviation[mask_3 & mask_4] = np.abs(investigation_data[f"{property}_propssi"] - investigation_data[f"{property}_giljarhus"])[mask_3 & mask_4] / np.abs(investigation_data[f"{property}_propssi"])[mask_3 & mask_4]
    deviation[mask_2 & mask_3] = np.abs(investigation_data[f"{property}_giljarhus"] - investigation_data[f"{property}_propssi"])[mask_2 & mask_3] / np.abs(investigation_data[f"{property}_giljarhus"] + np.full_like(investigation_data[f"{property}_propssi"], 1e-6))[mask_2 & mask_3]
    deviation[mask_1 & mask_4] = np.abs(investigation_data[f"{property}_propssi"] - investigation_data[f"{property}_giljarhus"])[mask_1 & mask_4] / np.abs(investigation_data[f"{property}_propssi"] + np.full_like(investigation_data[f"{property}_giljarhus"], 1e-6))[mask_1 & mask_4]
    deviation[mask_1 & mask_2] = np.zeros_like(deviation[mask_1 & mask_2])  # both quantities are very small, avoid division by zero. Set deviation to zero.

    
    
    # if np.any(np.abs(investigation_data[f"{property}_propssi"]) < 1e-8) and np.any(np.abs(investigation_data[f"{property}_giljarhus"]) < 1e-8):
    #     print(f"Warning: Both PropsSI and Giljarhus values for {property} are very small. Setting deviation to zero to avoid division by zero.")
    #     # both quantities are very small, avoid division by zero. Set deviation to zero.
    #     deviation = np.zeros_like(investigation_data[f"{property}_propssi"])
    # elif np.any(np.abs(investigation_data[f"{property}_propssi"]) < 1e-8):
    #     # propsSI quantity is very small, avoid division by zero. Use Giljarhus as reference.
    #     deviation = np.abs(investigation_data[f"{property}_giljarhus"] - investigation_data[f"{property}_propssi"]) / np.abs(investigation_data[f"{property}_giljarhus"])
    # elif np.any(np.abs(investigation_data[f"{property}_giljarhus"]) < 1e-8):
    #     # Giljarhus quantity is very small, avoid division by zero. Use PropsSI as reference.
    #     deviation = np.abs(investigation_data[f"{property}_propssi"] - investigation_data[f"{property}_giljarhus"]) / np.abs(investigation_data[f"{property}_propssi"])
    # else:
    #     # base case
    #     deviation = np.abs(investigation_data[f"{property}_propssi"] - investigation_data[f"{property}_giljarhus"]) / np.abs(investigation_data[f"{property}_propssi"])

    # find coordinates of 10 most deviating points
    indices = np.argsort(deviation)[-10:]
    print(f"Top 10 deviating points for {property}:")
    for idx in indices:
        print(f"  ST: {investigation_data['ST_coords'][idx]}, PropsSI: {investigation_data[f'{property}_propssi'][idx]}, Giljarhus: {investigation_data[f'{property}_giljarhus'][idx]}, Deviation: {deviation[idx]}")
    fig = thermoplot_cached(input_file_path)
    axes = fig.get_axes()
    scatter = axes[0].scatter(
        investigation_data["ST_coords"][:, 0],
        investigation_data["ST_coords"][:, 1],
        c=deviation,
        cmap='viridis',
        s=5,
        zorder=9,
    )
    fig.colorbar(scatter, ax=axes[0], label=f'Relative Deviation {property}')
    axes[0].set_xlabel('Entropy (S) [J/kg-K]')
    axes[0].set_ylabel('Temperature (T) [K]')
    plt.show()

    if plot_values:
        fig = thermoplot_cached(input_file_path)
        axes = fig.get_axes()
        scatter = axes[0].scatter(
            investigation_data["ST_coords"][:, 0],
            investigation_data["ST_coords"][:, 1],
            c=investigation_data[f"{property}_propssi"],
            cmap='viridis',
            s=5,
            zorder=9,
        )
        fig.colorbar(scatter, ax=axes[0], label=f'{property} PropsSI values')
        axes[0].set_xlabel('Entropy (S) [J/kg-K]')
        axes[0].set_ylabel('Temperature (T) [K]')
        plt.show()

        fig = thermoplot_cached(input_file_path)
        axes = fig.get_axes()
        scatter = axes[0].scatter(
            investigation_data["ST_coords"][:, 0],
            investigation_data["ST_coords"][:, 1],
            c=investigation_data[f"{property}_giljarhus"],
            cmap='viridis',
            s=5,
            zorder=9,
        )
        fig.colorbar(scatter, ax=axes[0], label=f'{property} Giljarhus values')
        axes[0].set_xlabel('Entropy (S) [J/kg-K]')
        axes[0].set_ylabel('Temperature (T) [K]')
        plt.show()





### Old Giljarhus system solver.


# # Instantiate guess history lists to track convergence of the iterative process.
# T_hist = [np.zeros_like(T_guess)]
# rho_V_hist = [np.zeros_like(rho_V_guess)]
# rho_L_hist = [np.zeros_like(rho_L_guess)]
# alpha_hist = [np.zeros_like(rho_values)]

# # Loop 1: T
# # loop 2: rho_v
# # Loop 3: rho_l
# while np.any(np.abs(T_guess - T_hist[-1]) > 1e-5):
#     T_hist.append(deepcopy(T_guess))
#     alpha_hist.append(deepcopy(alpha))
#     while np.any(np.abs(rho_V_guess - rho_V_hist[-1]) > 1e-5):
#         rho_V_hist.append(deepcopy(rho_V_guess))
#         while np.any(np.abs(rho_L_guess - rho_L_hist[-1]) > 1e-5):
#             rho_L_hist.append(deepcopy(rho_L_guess))
#             P_L = AS.PropsSI("P", "T", T_guess, "D", rho_L_guess)
#             P_V = AS.PropsSI("P", "T", T_guess, "D", rho_V_guess)
#             rho_L_guess[P_L > P_V] -= (np.abs(P_L[P_L > P_V] - P_V[P_L > P_V]))
#             rho_L_guess[P_L < P_V] += (np.abs(P_L[P_L < P_V] - P_V[P_L < P_V]))
#         G_L = AS.PropsSI("G", "T", T_guess, "D", rho_L_guess)
#         G_V = AS.PropsSI("G", "T", T_guess, "D", rho_V_guess) 
#         rho_V_guess[G_L > G_V] -= (np.abs(G_L[G_L > G_V] - G_V[G_L > G_V]))
#         rho_V_guess[G_L < G_V] += (np.abs(G_L[G_L < G_V] - G_V[G_L < G_V]))
#     alpha = (rho_values - rho_L_hist[-1]) / (rho_V_hist[-1] - rho_L_hist[-1])
#     LHS = (alpha * rho_V_hist[-1] * AS.PropsSI("U", "T", T_guess, "D", rho_V_hist[-1]) + \
#         (1 - alpha) * rho_L_hist[-1] * AS.PropsSI("U", "T", T_guess, "D", rho_L_hist[-1]))
#     RHS = rho_values * u_values
#     T_guess[LHS > RHS] -= (np.abs(LHS[LHS > RHS] - RHS[LHS > RHS]))
#     T_guess[LHS < RHS] += (np.abs(LHS[LHS < RHS] - RHS[LHS < RHS]))