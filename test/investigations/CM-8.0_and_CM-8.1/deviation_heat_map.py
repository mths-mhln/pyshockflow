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

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from fluid_properties.coolprop_interface import CoolPropAbstractState_v2
from thermoplot.isolines import construct_saturation_dome
from thermoplot.thermoplot import thermoplot_cached
from thermoplot.configthermoplot import ConfigThermoplot


# fluid property extraction method
AS = CoolPropAbstractState_v2("REFPROP", "R1234ze(E)")

# thermoplot saturation dome calculation scheme
input_file_path = "config/R1234ze(E).ini"
config = ConfigThermoplot(config_file=input_file_path)
config.get_thermoplot_settings()
dome_coords = construct_saturation_dome(config, AS)

# compute critical point and split saturation dome into LHS and RHS of crit point.
T_crit = AS.PropsSI("Tcrit")
crit_idx = np.argmin(np.abs(dome_coords[:, 1] - T_crit))
saturation_dome_LHS = dome_coords[:crit_idx + 1, :]
saturation_dome_RHS = dome_coords[crit_idx:, :]

# interpolate LHS and RHS to get S min and max values per T linspace value.
f_LHS = interp1d(saturation_dome_LHS[:, 1], saturation_dome_LHS[:, 0], kind='linear', fill_value='extrapolate')
f_RHS = interp1d(saturation_dome_RHS[:, 1], saturation_dome_RHS[:, 0], kind='linear', fill_value='extrapolate')

# linspace of T between T_min and T_max of saturation dome
T_min = min(dome_coords[:, 1])
T_max = max(dome_coords[:, 1])
T_values = np.linspace(T_min, T_max, 200)

# find largest S diff present in the saturation dome:
S_diffs = []
for T in T_values:
    S_min = f_LHS(T)
    S_max = f_RHS(T)
    S_diffs.append(S_max - S_min)
max_S_diff = max(S_diffs)

# initiate dictionary in which to store the investigation data
investigation_data = {
    "ST_coords":None,
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

# per T, linspace-like S values at which to evaluate rho, u
for i, T in enumerate(T_values):
    print(f"Processing T value {i+1}/{len(T_values)}: T = {T:.2f} K")
    S_min = f_LHS(T)
    S_max = f_RHS(T)
    S_diff = S_max - S_min
    S_values = np.linspace(S_min, S_max, int(100 * S_diff / max_S_diff))  # scale number of points based on S_diff
    if S_values.size == 0:
        continue
    T_array = np.full_like(S_values, T)  # create an array of T values corresponding to S_values
    coords = np.column_stack((S_values, T_array))  # combine S and T into a single array of coordinates

    # generation of dictionary saving coordinate on T-S (the T, S pair) and associated rho, u values using PropsSI
    rho_values = np.array([AS.PropsSI("D", "T", T, "S", S) for S in S_values])
    u_values = np.array([AS.PropsSI("U", "T", T, "S", S) for S in S_values])

    # compute entropy values using PropsSI, and sanity check that these are the same S as the coordinate S values
    S_propssi = np.array([AS.PropsSI("S", "U", u, "D", rho) for u, rho in zip(u_values, rho_values)])
    assert np.allclose(S_propssi, S_values), "Mismatch between computed S and coordinate S values using PropsSI."

    # compute entropy using Giljarhus system of equations, saving to dictionary
    T_guess = np.full_like(rho_values, T)
    rho_V_guess = AS.PropsSI("D", "T", T_guess, "Q", np.ones_like(rho_values)) 
    rho_L_guess = AS.PropsSI("D", "T", T_guess, "Q", np.zeros_like(rho_values))

    T_hist = [np.zeros_like(T_guess)]
    rho_V_hist = [np.zeros_like(rho_V_guess)]
    rho_L_hist = [np.zeros_like(rho_L_guess)]
    alpha_hist = [np.zeros_like(rho_values)]

    # Loop 1: T
    # loop 2: rho_v
    # Loop 3: rho_l
    while np.all(np.abs(T_guess - T_hist[-1])) > 1e-3:
        while np.all(np.abs(rho_V_guess - rho_V_hist[-1]))> 1e-3:
            while np.all(np.abs(rho_L_guess - rho_L_hist[-1])) > 1e-3:
                P_L = AS.PropsSI("P", "T", T_guess, "D", rho_L_guess)
                P_V = AS.PropsSI("P", "T", T_guess, "D", rho_V_guess)
                rho_L_guess[P_L > P_V] -= (np.abs(P_L[P_L > P_V] - P_V[P_L > P_V]))
                rho_L_guess[P_L < P_V] += (np.abs(P_L[P_L < P_V] - P_V[P_L < P_V]))
                rho_L_hist.append(rho_L_guess)
            G_L = AS.PropsSI("G", "T", T_guess, "D", rho_L_guess)
            G_V = AS.PropsSI("G", "T", T_guess, "D", rho_V_guess)  
            rho_V_guess[G_L > G_V] -= (np.abs(G_L[G_L > G_V] - G_V[G_L > G_V]))
            rho_V_guess[G_L < G_V] += (np.abs(G_L[G_L < G_V] - G_V[G_L < G_V]))
            rho_V_hist.append(rho_V_guess)
        alpha = (rho_values - rho_L_hist[-1]) / (rho_V_hist[-1] - rho_L_hist[-1])
        LHS = (alpha * rho_V_hist[-1] * AS.PropsSI("U", "T", T_guess, "D", rho_V_hist[-1]) + \
               (1 - alpha) * rho_L_hist[-1] * AS.PropsSI("U", "T", T_guess, "D", rho_L_hist[-1]))
        RHS = rho_values * u_values
        T_guess[LHS > RHS] -= (np.abs(LHS[LHS > RHS] - RHS[LHS > RHS]))
        T_guess[LHS < RHS] += (np.abs(LHS[LHS < RHS] - RHS[LHS < RHS]))
        T_hist.append(T_guess)
        alpha_hist.append(alpha)

    print("rho_values = ", rho_values)
    print("System solution: ", T_guess, rho_V_guess, rho_L_guess, alpha)
    
    # calculate properties of interest to observe deviation

    # entropy
    S_V = AS.PropsSI("S", "T", T_hist[-1], "Q", 1)
    S_L = AS.PropsSI("S", "T", T_hist[-1], "Q", 0)
    Q = alpha_hist[-1] * AS.PropsSI("D", "T", T_array, "Q", 1) / rho_values
    S_giljarhus = Q * S_V + (1 - Q) * S_L

    # pressure
    P_V = AS.PropsSI("P", "T", T_hist[-1], "Q", 1)
    P_L = AS.PropsSI("P", "T", T_hist[-1], "Q", 0)
    P_giljarhus = alpha_hist[-1] * P_V + (1 - alpha_hist[-1]) * P_L
    P_propssi = np.array([AS.PropsSI("P", "U", u, "D", rho) for u, rho in zip(u_values, rho_values)])

    # density
    rho_V = AS.PropsSI("D", "T", T_hist[-1], "Q", 1)
    rho_L = AS.PropsSI("D", "T", T_hist[-1], "Q", 0)
    rho_giljarhus = alpha_hist[-1] * rho_V + (1 - alpha_hist[-1]) * rho_L
    rho_propssi = rho_values

    # static internal energy
    U_V = AS.PropsSI("U", "T", T_hist[-1], "Q", 1)
    U_L = AS.PropsSI("U", "T", T_hist[-1], "Q", 0)
    U_giljarhus = Q * U_V + (1 - Q) * U_L
    U_propssi = np.array([AS.PropsSI("U", "P", p, "D", rho) for p, rho in zip(P_propssi, rho_values)])

    # volume fraction
    alpha_giljarhus = alpha_hist[-1]
    Q_propssi = np.array([AS.PropsSI("Q", "U", u, "D", rho) for u, rho in zip(u_values, rho_values)])
    print("Q_propssi = ", Q_propssi)
    alpha_propssi = Q_propssi * rho_values / AS.PropsSI("D", "T", T_array, "Q", 1)
    print("alpha_propssi = ", alpha_propssi)
    print("alpha_giljarhus = ", alpha_giljarhus)

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

# generate heat map showing the relative deviation from PropsSI and Giljarhus system of equations.
deviation = np.abs(investigation_data["S_propssi"] - investigation_data["S_giljarhus"]) / np.abs(investigation_data["S_propssi"])
# map all deviation values > 0.3 to 0.3 for better visualization
fig = thermoplot_cached(input_file_path)
axes = fig.get_axes()
# deviation[deviation > 0.2] = 0.2
scatter = axes[0].scatter(
    investigation_data["ST_coords"][:, 0],
    investigation_data["ST_coords"][:, 1],
    c=deviation,
    cmap='viridis',
    s=5,
    zorder=9,
)
# im = axes[0].tripcolor(investigation_data["ST_coords"][:, 0], investigation_data["ST_coords"][:, 1], np.abs(investigation_data["S_propssi"] - investigation_data["S_giljarhus"]) / np.abs(investigation_data["S_propssi"]), cmap="viridis", rasterized=True, vmin=-4, vmax=4)
# # axes[0].scatter(investigation_data["ST_coords"][:, 0], investigation_data["ST_coords"][:, 1], c=np.abs(investigation_data["S_propssi"] - investigation_data["S_giljarhus"]) / np.abs(investigation_data["S_propssi"]), cmap='viridis', s=5, zorder=9)
fig.colorbar(scatter, ax=axes[0], label=r'Relative Deviation $\frac{|S_{propssi} - S_{gijarhus}|}{|S_{propssi}|}$')
axes[0].set_xlabel('Entropy (S) [J/kg-K]')
axes[0].set_ylabel('Temperature (T) [K]')
plt.show()


# generate heat map showing the relative deviation from PropsSI and Giljarhus system of equations.
deviation = np.abs(investigation_data["P_propssi"] - investigation_data["P_giljarhus"]) / np.abs(investigation_data["P_propssi"])
# map all deviation values > 0.3 to 0.3 for better visualization
fig = thermoplot_cached(input_file_path)
axes = fig.get_axes()
# deviation[deviation > 0.2] = 0.2
scatter = axes[0].scatter(
    investigation_data["ST_coords"][:, 0],
    investigation_data["ST_coords"][:, 1],
    c=deviation,
    cmap='viridis',
    s=5,
    zorder=9,
)
# im = axes[0].tripcolor(investigation_data["ST_coords"][:, 0], investigation_data["ST_coords"][:, 1], np.abs(investigation_data["P_propssi"] - investigation_data["P_giljarhus"]) / np.abs(investigation_data["P_propssi"]), cmap="viridis", rasterized=True, vmin=-4, vmax=4)
# # axes[0].scatter(investigation_data["ST_coords"][:, 0], investigation_data["ST_coords"][:, 1], c=np.abs(investigation_data["P_propssi"] - investigation_data["P_giljarhus"]) / np.abs(investigation_data["P_propssi"]), cmap='viridis', s=5, zorder=9)
fig.colorbar(scatter, ax=axes[0], label=r'Relative Deviation $\frac{|P_{propssi} - P_{gijarhus}|}{|P_{propssi}|}$')
axes[0].set_xlabel('Entropy (S) [J/kg-K]')
axes[0].set_ylabel('Temperature (T) [K]')
plt.show()


# generate heat map showing the relative deviation from PropsSI and Giljarhus system of equations.
deviation = np.abs(investigation_data["U_propssi"] - investigation_data["U_giljarhus"]) / np.abs(investigation_data["U_propssi"])
# map all deviation values > 0.3 to 0.3 for better visualization
fig = thermoplot_cached(input_file_path)
axes = fig.get_axes()
# deviation[deviation > 0.2] = 0.2
scatter = axes[0].scatter(
    investigation_data["ST_coords"][:, 0],
    investigation_data["ST_coords"][:, 1],
    c=deviation,
    cmap='viridis',
    s=5,
    zorder=9,
)
# im = axes[0].tripcolor(investigation_data["ST_coords"][:, 0], investigation_data["ST_coords"][:, 1], np.abs(investigation_data["U_propssi"] - investigation_data["U_giljarhus"]) / np.abs(investigation_data["U_propssi"]), cmap="viridis", rasterized=True, vmin=-4, vmax=4)
# # axes[0].scatter(investigation_data["ST_coords"][:, 0], investigation_data["ST_coords"][:, 1], c=np.abs(investigation_data["U_propssi"] - investigation_data["U_giljarhus"]) / np.abs(investigation_data["U_propssi"]), cmap='viridis', s=5, zorder=9)
fig.colorbar(scatter, ax=axes[0], label=r'Relative Deviation $\frac{|U_{propssi} - U_{gijarhus}|}{|U_{propssi}|}$')
axes[0].set_xlabel('Internal Energy (U) [J/kg]')
axes[0].set_ylabel('Temperature (T) [K]')
plt.show()


# generate heat map showing the relative deviation from PropsSI and Giljarhus system of equations.
deviation = np.abs(investigation_data["alpha_propssi"] - investigation_data["alpha_giljarhus"]) / np.abs(investigation_data["alpha_propssi"])
print(investigation_data["alpha_propssi"])
print(investigation_data["alpha_giljarhus"])
print(investigation_data["alpha_propssi"] - investigation_data["alpha_giljarhus"])
# map all deviation values > 0.3 to 0.3 for better visualization
fig = thermoplot_cached(input_file_path)
axes = fig.get_axes()
# deviation[deviation > 0.2] = 0.2
scatter = axes[0].scatter(
    investigation_data["ST_coords"][:, 0],
    investigation_data["ST_coords"][:, 1],
    c=deviation,
    cmap='viridis',
    s=5,
    zorder=9,
)
# im = axes[0].tripcolor(investigation_data["ST_coords"][:, 0], investigation_data["ST_coords"][:, 1], np.abs(investigation_data["alpha_propssi"] - investigation_data["alpha_giljarhus"]) / np.abs(investigation_data["alpha_propssi"]), cmap="viridis", rasterized=True, vmin=-4, vmax=4)
# # axes[0].scatter(investigation_data["ST_coords"][:, 0], investigation_data["ST_coords"][:, 1], c=np.abs(investigation_data["alpha_propssi"] - investigation_data["alpha_giljarhus"]) / np.abs(investigation_data["alpha_propssi"]), cmap='viridis', s=5, zorder=9)
fig.colorbar(scatter, ax=axes[0], label=r'Relative Deviation $\frac{|alpha_{propssi} - alpha_{giljarhus}|}{|alpha_{propssi}|}$')
axes[0].set_xlabel('Internal Energy (U) [J/kg]')
axes[0].set_ylabel('Temperature (T) [K]')
plt.show()


# generate heat map showing the relative deviation from PropsSI and Giljarhus system of equations.
deviation = np.abs(investigation_data["rho_propssi"] - investigation_data["rho_giljarhus"]) / np.abs(investigation_data["rho_propssi"])
print(investigation_data["rho_propssi"])
print(investigation_data["rho_giljarhus"])
print(investigation_data["rho_propssi"] - investigation_data["rho_giljarhus"])
# map all deviation values > 0.3 to 0.3 for better visualization
fig = thermoplot_cached(input_file_path)
axes = fig.get_axes()
# deviation[deviation > 0.2] = 0.2
scatter = axes[0].scatter(
    investigation_data["ST_coords"][:, 0],
    investigation_data["ST_coords"][:, 1],
    c=deviation,
    cmap='viridis',
    s=5,
    zorder=9,
)
# im = axes[0].tripcolor(investigation_data["ST_coords"][:, 0], investigation_data["ST_coords"][:, 1], np.abs(investigation_data["rho_propssi"] - investigation_data["rho_giljarhus"]) / np.abs(investigation_data["rho_propssi"]), cmap="viridis", rasterized=True, vmin=-4, vmax=4)
# # axes[0].scatter(investigation_data["ST_coords"][:, 0], investigation_data["ST_coords"][:, 1], c=np.abs(investigation_data["rho_propssi"] - investigation_data["rho_giljarhus"]) / np.abs(investigation_data["rho_propssi"]), cmap='viridis', s=5, zorder=9)
fig.colorbar(scatter, ax=axes[0], label=r'Relative Deviation $\frac{|rho_{propssi} - rho_{giljarhus}|}{|rho_{propssi}|}$')
axes[0].set_xlabel('Density (rho) [kg/m³]')
axes[0].set_ylabel('Temperature (T) [K]')
plt.show()







