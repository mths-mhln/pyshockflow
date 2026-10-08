
from fluid_properties.coolprop_interface import CoolPropAbstractState_v2
from pyshockflow.fluid import FluidReal
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import pickle

# AS = CoolPropAbstractState_v2("REFPROP", "R1234ze(E)")
# FluidRealObj = FluidReal("R1234ze(E)", "REFPROP", "abstractstate_v2")

# p = AS.PropsSI("P", "S", 1600.72573613, "T", 382.34715144)  
# rho = AS.PropsSI("D", "S", 1600.72573613, "T", 382.34715144)
# print("p: ", p)
# print("rho: ", rho)

# SOS = FluidRealObj.computeSoundSpeed_p_rho(p, rho)
# print(SOS)

# print(FluidRealObj.computeSoundSpeed_p_rho(3622983.93194916, 409.35743802))

# print(FluidRealObj.computeSoundSpeed_p_rho(3622983.931949161, 409.3574380247973))

# print(FluidRealObj.computeSoundSpeed_p_rho(np.array([3634864.9992578067]), np.array([489.2384830777358])))



# with open("Results/lettieri/output_L5_friction/step_001000.pik", "rb") as f:
#     data = pickle.load(f)

# print(data["fluidState"]["Pressure"])



import pickle
with open("Results/berana/output_B1_1600_09/Results.pik", "rb") as file:
    a = pickle.load(file)

with open("Results/berana/output_B1/Results.pik", "rb") as file:
    b = pickle.load(file)


# fluidState = {}
# for key in a["fluidStateHistory"].keys():

#     fluidState[key] = a["fluidStateHistory"][key][:, -1]  




# #Interpolate values to a new mesh
# from scipy.interpolate import interp1d

# xMeshNodes_old = a["meshData"]["xMeshNodes"]
# x = np.linspace(xMeshNodes_old[0], xMeshNodes_old[-1], 1602)  # New mesh with 402 nodes (including ghost nodes)

# for key in fluidState.keys():
#     interp_func = interp1d(xMeshNodes_old, fluidState[key], kind='linear', fill_value="extrapolate")
#     fluidState[key] = interp_func(x)

a_fluidState = {}
for key in a["fluidStateHistory"].keys():
    a_fluidState[key] = a["fluidStateHistory"][key][:, -1]

b_fluidState = {}
for key in b["fluidStateHistory"].keys():
    b_fluidState[key] = b["fluidStateHistory"][key][:, -1]

# compute max abs and rel difference between a_fluidState and b_fluidState
for key in a_fluidState.keys():
    if key in b_fluidState:
        max_diff = np.max(np.abs(a_fluidState[key] - b_fluidState[key]))
        rel_diff = max_diff / np.max(np.abs(a_fluidState[key]))
        print(f"Max abs difference in {key}: {max_diff}")
        print(f"Max rel difference in {key}: {rel_diff}")



import pickle
import numpy as np

base = "Results/berana/output_B1_{n}_09/Results.pik"
meshSizes = [200, 400, 800, 1600]

def loadFinalState(n):
    with open(base.format(n=n), "rb") as file:
        r = pickle.load(file)
    state = {k: v[:, -1] for k, v in r["fluidStateHistory"].items()}
    return r["meshData"]["xMeshNodes"], state

results = {n: loadFinalState(n) for n in meshSizes}

for nCoarse, nFine in zip(meshSizes[:-1], meshSizes[1:]):
    xCoarse, sCoarse = results[nCoarse]
    xFine,   sFine   = results[nFine]

    print(f"\n{nCoarse} -> {nFine}")
    relDiffs = []
    for key in sFine:
        # Interpolate the coarse solution onto the fine interior nodes (halo nodes excluded).
        coarseOnFine = np.interp(xFine[1:-1], xCoarse[1:-1], sCoarse[key][1:-1])
        absDiff = np.max(np.abs(sFine[key][1:-1] - coarseOnFine))
        relDiff = absDiff / (np.max(np.abs(sFine[key][1:-1])) + 1e-300)
        relDiffs.append(relDiff)
        print(f"  {key:22s} max abs: {absDiff:.6e}   max rel: {relDiff:.6e}")
    print(f"  max rel over all variables: {max(relDiffs):.6e}")