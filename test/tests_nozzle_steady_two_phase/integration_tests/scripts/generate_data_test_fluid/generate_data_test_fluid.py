import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import copy
import pickle
from scipy.optimize import newton

from pyshockflow.fluid import FluidReal
from fluid_properties.coolprop_interface import CoolPropAbstractState_v2

from thermoplot.isolines import construct_saturation_dome
from thermoplot.thermoplot import thermoplot_cached
from thermoplot.configthermoplot import ConfigThermoplot



# prepare saturation_dome_testing data
# Data generated using current code implementation. Verification performed by inspection.
fluid_data = {
    "CO2": {
        "config_file": "config/CO2.ini",
        "data_file_path": "../../data/test_fluid/data_test_fluid_CO2.pkl",
        "isentropic_expansion_config": {
            "T_start": 288,
            "T_end": 240,
            "S": 1060
        }, 
        "isothermal_process_config": {
            "T": 250,
            "S_start": 1200,
            "S_end": 2100
        }
    },
    "R1234ze(E)": {
        "config_file": "config/R1234ze(E).ini",
        "data_file_path": "../../data/test_fluid/data_test_fluid_R1234ze(E).pkl",
        "isentropic_expansion_config": {
            "T_start": 288,
            "T_end": 240,
            "S": 1060
        }, 
        "isothermal_process_config": {
            "T": 250,
            "S_start": 1200,
            "S_end": 2100
        }
    }
}

for fluid in fluid_data.keys():
    # instantiate fluid_real object whose method, computeSoundSpeed_p_rho, will be tested
    fluid_real_obj = FluidReal(fluid, "REFPROP", "abstractstate_v2")

    # extract thermodynamic dome (T, S) coordinates for the fluids the method will be 
    # tested on. The thermoplot built-in functionality is used for this purpose. 
    # for more information on how to use thermoplot, please refer to the source code included
    # in this project folder.
    input_file_path = fluid_data[fluid]["config_file"]
    fig = thermoplot_cached(input_file_path)
    axes = fig.get_axes()
    AS = CoolPropAbstractState_v2("REFPROP", fluid)
    config = ConfigThermoplot(config_file=input_file_path)
    config.get_thermoplot_settings()
    dome_coords = construct_saturation_dome(config, AS)

    # instantiate dictionary containing the verification data for the five test cases. 
    data_dict = dict.fromkeys(
        ["isentropic_expansion", 
        "isothermal_process", 
        "parallel_plus_001", 
        "parallel_minus_001", 
        "parallel_000"], 
        None)
    data_dict["isentropic_expansion"] = dict.fromkeys(["thdy_coords", "sound_speed"], None)
    data_dict["isothermal_process"] = dict.fromkeys(["thdy_coords", "sound_speed"], None)
    data_dict["parallel_plus_001"] = dict.fromkeys(["thdy_coords", "sound_speed"], None)
    data_dict["parallel_minus_001"] = dict.fromkeys(["thdy_coords", "sound_speed"], None)
    data_dict["parallel_000"] = dict.fromkeys(["thdy_coords", "sound_speed"], None)


    # build isentropic expansion case for fluid:
    T = np.linspace(fluid_data[fluid]["isentropic_expansion_config"]["T_start"], 
                    fluid_data[fluid]["isentropic_expansion_config"]["T_end"], 100)
    S = np.ones_like(T) * fluid_data[fluid]["isentropic_expansion_config"]["S"]
    P = AS.PropsSI("P", "T", T, "S", S)
    rho = AS.PropsSI("D", "T", T, "S", S)
    sound_speeds = np.array([fluid_real_obj.computeSoundSpeed_p_rho(p, rho) for p, rho in zip(P, rho)])
    thdy_coords = np.column_stack((S, T))
    data_dict["isentropic_expansion"]["thdy_coords"] = thdy_coords
    data_dict["isentropic_expansion"]["sound_speed"] = sound_speeds

    # Build the isothermal process case for fluid:
    S = np.linspace(fluid_data[fluid]["isothermal_process_config"]["S_start"], 
                    fluid_data[fluid]["isothermal_process_config"]["S_end"], 100)
    T = np.ones_like(S) * fluid_data[fluid]["isothermal_process_config"]["T"]
    P = AS.PropsSI("P", "T", T, "S", S)
    rho = AS.PropsSI("D", "T", T, "S", S)
    sound_speeds = np.array([fluid_real_obj.computeSoundSpeed_p_rho(p, rho) for p, rho in zip(P, rho)])
    thdy_coords = np.column_stack((S, T))
    data_dict["isothermal_process"]["thdy_coords"] = thdy_coords
    data_dict["isothermal_process"]["sound_speed"] = sound_speeds

    # build the parallel plus 001 case for fluid
    # the parallel plus 001 case is a set of thdy states running parallel to the saturation dome 
    # at +001% it's temperature value.
    dome_coords_3 = copy.deepcopy(dome_coords)
    dome_coords_3[:, 1] *= 1.00001
    P = AS.PropsSI("P", "T", dome_coords_3[:, 1], "S", dome_coords_3[:, 0])
    rho = AS.PropsSI("D", "T", dome_coords_3[:, 1], "S", dome_coords_3[:, 0])
    sound_speeds = np.array([fluid_real_obj.computeSoundSpeed_p_rho(p, rho) for p, rho in zip(P, rho)])
    thdy_coords = np.column_stack((dome_coords_3[:, 0], dome_coords_3[:, 1]))
    data_dict["parallel_plus_001"]["thdy_coords"] = thdy_coords
    data_dict["parallel_plus_001"]["sound_speed"] = sound_speeds

    # build the parallel minus 001 case for fluid
    # the parallel minus 001 case is a set of thdy states running parallel to the saturation dome 
    # at -001% it's temperature value.
    dome_coords_4 = copy.deepcopy(dome_coords)
    dome_coords_4[:, 1] *= 0.9999
    P = AS.PropsSI("P", "T", dome_coords_4[:, 1], "S", dome_coords_4[:, 0])
    rho = AS.PropsSI("D", "T", dome_coords_4[:, 1], "S", dome_coords_4[:, 0])
    sound_speeds = np.array([fluid_real_obj.computeSoundSpeed_p_rho(p, rho) for p, rho in zip(P, rho)])
    thdy_coords = np.column_stack((dome_coords_4[:, 0], dome_coords_4[:, 1]))
    data_dict["parallel_minus_001"]["thdy_coords"] = thdy_coords
    data_dict["parallel_minus_001"]["sound_speed"] = sound_speeds

    # build the parallel 000 case for fluid
    # the parallel 000 case is a set of thdy states running parallel to the saturation dome 
    # at exactly it's temperature value.
    dome_coords_5 = copy.deepcopy(dome_coords)
    dome_coords_5[:, 1] *= 1.0
    P = AS.PropsSI("P", "T", dome_coords_5[:, 1], "S", dome_coords_5[:, 0])
    rho = AS.PropsSI("D", "T", dome_coords_5[:, 1], "S", dome_coords_5[:, 0])
    sound_speeds = np.array([fluid_real_obj.computeSoundSpeed_p_rho(p, rho) for p, rho in zip(P, rho)])
    thdy_coords = np.column_stack((dome_coords_5[:, 0], dome_coords_5[:, 1]))
    data_dict["parallel_000"]["thdy_coords"] = thdy_coords
    data_dict["parallel_000"]["sound_speed"] = sound_speeds
    print(data_dict["parallel_000"]["sound_speed"])
    
    # save dict as pkl file
    with open(fluid_data[fluid]["data_file_path"], "wb") as f:
        pickle.dump(data_dict, f)



# prepare benchmark_of_the_dem_data:
df = pd.read_csv("data/HEM_sos_benchmark_of_dem_fig_1b.csv")
data = dict.fromkeys(df.columns)
for i, col in enumerate(df.columns):
    data[col] = df.iloc[1:, i].values


AS = CoolPropAbstractState_v2("REFPROP", "Water")
fluid_real_obj = FluidReal("Water", "REFPROP", "abstractstate_v2")
# # Test requires some additional calculations to convert quality into void fraction
# alpha_V = data["alpha_V"]
# # the next parameters are known from how the test case was constructed.
# p = np.ones_like(alpha_V) * 0.1e6  # 0.1 MPa
# rho_v = AS.PropsSI("D", "P", p, "Q", np.ones_like(alpha_V))
# print(rho_v)

# # solve equation iteratively for quality
# def func(Q, alpha_v, rho_v, p):
#     if Q>= 1 or Q <= 0:
#         return np.inf  # Return a large number to indicate an invalid solution
#     else:
#         return (alpha_v / Q) * rho_v - AS.PropsSI("D", "P", p, "Q", Q)

# print("optimizing")
# for alpha_v_val, rho_v_val, p_val in zip(alpha_V, rho_v, p):
#     Q = newton(func, x0=1e-6, args=(alpha_v_val, rho_v_val, p_val))
#     print(f"alpha_v: {alpha_v_val}, rho_v: {rho_v_val}, Q: {Q}")

# # plot variation of function with Q
# Q = np.linspace(0.000001, 0.999, len(alpha_V))
# func_values = [func(Q_val, alpha_V[0], rho_v[0], p[0]) for Q_val in Q]
# # print(func_values)
# # print(alpha_V[0], rho_v[0])
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(Q, func_values)
# ax.set_xlabel("Quality (Q)")   
# ax.set_ylabel("Function Value")
# ax.set_title("Variation of Function with Quality (Q)")
# ax.grid()
# plt.show()

# Test requires some additional calculations to convert quality into void fraction
alpha_V = data["alpha_V"]
# the next parameters are known from how the test case was constructed.
p = np.ones_like(alpha_V) * 0.1e6  # 0.1 MPa
rho_v = AS.PropsSI("D", "P", p, "Q", np.ones_like(alpha_V))

# solve equation iteratively for rho
def func(Q, alpha_v, rho_v, p):
    if np.any(Q < 0) or np.any(Q > 1):
        return np.inf  # Return a large number to indicate an invalid solution
    else:
        return (AS.PropsSI("D", "P", p, "Q", Q)/rho_v)*Q - alpha_v

Q = newton(func, x0=np.full_like(alpha_V, 1e-6),
        args=(alpha_V, rho_v, p))
print("obtained Q", Q)

rho = AS.PropsSI("D", "P", p, "Q", Q)

# compute the sound speed using the FluidReal method
soundSpeed_HEM = fluid_real_obj.computeSoundSpeed_p_rho(p, rho)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(alpha_V, soundSpeed_HEM, label="Computed Sound Speed (HEM)")
ax.plot(alpha_V, data["HEM_sound_speed"], label="Expected Sound Speed (De Lorenzo)")
ax.set_xlabel("Void Fraction (alpha_V)")
ax.set_ylabel("Sound Speed (m/s)")
ax.legend()
plt.show()

with open("../../data/test_fluid/HEM_sos_benchmark_of_dem_fig_1b.pkl", "wb") as f:
    pickle.dump(data, f)
