import sys
import copy

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from pyshockflow.fluid import FluidReal
from fluid_properties.coolprop_interface import CoolPropAbstractState_v2
from thermoplot.isolines import construct_saturation_dome
from thermoplot.thermoplot import thermoplot_cached
from thermoplot.configthermoplot import ConfigThermoplot


#########################################################
#   CM-7.2: mixture property p, T calls investigation
#########################################################
# Saturation dome testing
# =======================
# CM-7.2 investigation consists of 5 cases, investigating the ability of the 
# computeEntropy_p_T method of the FluidReal class to compute the fluid entropy
# for thdy states close to the saturation dome using p, T inputs. The resulting 
# values are compared to the entropy computed from rho, u inputs, which is the 
# input pair obtained after every iteration of the driver.py solve() function. 
# The five cases are:
# 1) isentropic expansion case: a set of thdy states with constant entropy 
#    expanding from subcooled fluid to the two -phase region at quality ~0.3
# 2) isothermal process case: a set of thdy states with constant temperature 
#    going form quality ~0.3 to superheated vapour. The case is isothermal rather
#    than isentropic due to the "dryness" of the working fluid, see Martin T. White
#    paper "Cycle and turbine optimisation for an ORC operating with two-phase expansion"
# 3) parallel plus 001 case: a set of thdy states running parallel to the saturation dome
#    at +001% it's temperature value.
# 4) parallel minus 001 case: a set of thdy states running parallel to the saturation dome 
#    at -001% it's temperature value.
# 5) parallel 000 case: a set of thdy states running parallel to the saturation dome
#    at exactly it's temperature value.
#
# the test aims to evaluate computeEntropy_p_T's ability to return sensible values
# close to the saturation dome. My hypothesis is that it will not be able to. If the 
# hypothesis is correct, then it can be expected that other p, T calls will also fail, 
# and a widespread replacement of p, T calls by rho, u calls will be necessary.
fluid_data = {
    "CO2": {
        "config_file": "config/CO2.ini",
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
        "isentropic_expansion_config": {
            "T_start": 288,
            "T_end": 240,
            "S": 1060
        }, 
        "isothermal_process_config": {
            "T": 300,
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
    U = AS.PropsSI("U", "T", T, "S", S)
    rho = AS.PropsSI("D", "T", T, "S", S)
    S_calculated_p_T = AS.PropsSI("S", "T", T, "P", P)
    S_calculated_rho_U = AS.PropsSI("S", "D", rho, "U", U)
    # create thermoplot for illustration of difference between S_calculated_p_T and S_calculated_rho_U
    input_file_path = fluid_data[fluid]["config_file"]
    fig = thermoplot_cached(input_file_path)
    axes = fig.get_axes()
    axes[0].scatter(S_calculated_p_T, T, label="S_calculated_p_T", color="blue", s=5)
    axes[0].scatter(S_calculated_rho_U, T, label="S_calculated_rho_U", color="red", s=5)
    plt.legend()
    plt.show()

    # Build the isothermal process case for fluid:
    S = np.linspace(fluid_data[fluid]["isothermal_process_config"]["S_start"], 
                    fluid_data[fluid]["isothermal_process_config"]["S_end"], 100)
    T = np.ones_like(S) * fluid_data[fluid]["isothermal_process_config"]["T"]
    P = AS.PropsSI("P", "T", T, "S", S)
    U = AS.PropsSI("U", "T", T, "S", S)
    rho = AS.PropsSI("D", "T", T, "S", S)
    S_calculated_p_T = AS.PropsSI("S", "T", T, "P", P)
    S_calculated_rho_U = AS.PropsSI("S", "D", rho, "U", U)
    # create thermoplot for illustration of difference between S_calculated_p_T and S_calculated_rho_U
    input_file_path = fluid_data[fluid]["config_file"]
    fig = thermoplot_cached(input_file_path)
    axes = fig.get_axes()
    axes[0].scatter(S_calculated_p_T, T, label="S_calculated_p_T", color="blue", s=5)
    axes[0].scatter(S_calculated_rho_U, T, label="S_calculated_rho_U", color="red", s=5)
    plt.legend()
    plt.show()

    # build the parallel plus 001 case for fluid
    # the parallel plus 001 case is a set of thdy states running parallel to the saturation dome 
    # at +001% it's temperature value.
    dome_coords_3 = copy.deepcopy(dome_coords)
    dome_coords_3[:, 1] *= 1.00001
    P = AS.PropsSI("P", "T", dome_coords_3[:, 1], "S", dome_coords_3[:, 0])
    U = AS.PropsSI("U", "T", dome_coords_3[:, 1], "S", dome_coords_3[:, 0])
    rho = AS.PropsSI("D", "T", dome_coords_3[:, 1], "S", dome_coords_3[:, 0])
    S_calculated_p_T = AS.PropsSI("S", "T", dome_coords_3[:, 1], "P", P)
    S_calculated_rho_U = AS.PropsSI("S", "D", rho, "U", U)
    # create thermoplot for illustration of difference between S_calculated_p_T and S_calculated_rho_U
    input_file_path = fluid_data[fluid]["config_file"]
    fig = thermoplot_cached(input_file_path)
    axes = fig.get_axes()
    axes[0].scatter(S_calculated_p_T, dome_coords_3[:, 1], label="S_calculated_p_T", color="blue", s=5)
    axes[0].scatter(S_calculated_rho_U, dome_coords_3[:, 1], label="S_calculated_rho_U", color="red", s=5)
    plt.legend()
    plt.show()

    # build the parallel minus 001 case for fluid
    # the parallel minus 001 case is a set of thdy states running parallel to the saturation dome 
    # at -001% it's temperature value.
    dome_coords_4 = copy.deepcopy(dome_coords)
    dome_coords_4[:, 1] *= 0.9999
    P = AS.PropsSI("P", "T", dome_coords_4[:, 1], "S", dome_coords_4[:, 0])
    U = AS.PropsSI("U", "T", dome_coords_4[:, 1], "S", dome_coords_4[:, 0])
    rho = AS.PropsSI("D", "T", dome_coords_4[:, 1], "S", dome_coords_4[:, 0])
    S_calculated_p_T = AS.PropsSI("S", "T", dome_coords_4[:, 1], "P", P)
    S_calculated_rho_U = AS.PropsSI("S", "D", rho, "U", U)
    # create thermoplot for illustration of difference between S_calculated_p_T and S_calculated_rho_U
    input_file_path = fluid_data[fluid]["config_file"]
    fig = thermoplot_cached(input_file_path)
    axes = fig.get_axes()
    axes[0].scatter(S_calculated_p_T, dome_coords_4[:, 1], label="S_calculated_p_T", color="blue", s=5)
    axes[0].scatter(S_calculated_rho_U, dome_coords_4[:, 1], label="S_calculated_rho_U", color="red", s=5)
    plt.legend()
    plt.show()

    # build the parallel 000 case for fluid
    # the parallel 000 case is a set of thdy states running parallel to the saturation dome 
    # at exactly it's temperature value.
    dome_coords_5 = copy.deepcopy(dome_coords)
    dome_coords_5[:, 1] *= 1.0
    P = AS.PropsSI("P", "T", dome_coords_5[:, 1], "S", dome_coords_5[:, 0])
    U = AS.PropsSI("U", "T", dome_coords_5[:, 1], "S", dome_coords_5[:, 0])
    rho = AS.PropsSI("D", "T", dome_coords_5[:, 1], "S", dome_coords_5[:, 0])
    S_calculated_p_T = AS.PropsSI("S", "T", dome_coords_5[:, 1], "P", P)
    S_calculated_rho_U = AS.PropsSI("S", "D", rho, "U", U)
    # create thermoplot for illustration of difference between S_calculated_p_T and S_calculated_rho_U
    input_file_path = fluid_data[fluid]["config_file"]
    fig = thermoplot_cached(input_file_path)
    axes = fig.get_axes()
    axes[0].scatter(S_calculated_p_T, dome_coords_5[:, 1], label="S_calculated_p_T", color="blue", s=5)
    axes[0].scatter(S_calculated_rho_U, dome_coords_5[:, 1], label="S_calculated_rho_U", color="red", s=5)
    plt.legend()
    plt.show()
    

