import numpy as np
import matplotlib.pyplot as plt
import pickle

from pathlib import Path
from pyshockflow.plot_styles import *
from scipy.optimize import fsolve

# steplist = ["0000"]
steplist = ["0000", "0250", "0500", "0750", "1000", "1250", "1260", "1280", "1300", "1320", "1340", "1360", "1380", "1400", "1420", "1440"]
pickleList = [f"Results/outletPressure_136kPa_NX_200/step_00{step}.pik" for step in steplist]


fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for i, pickleFile in enumerate(pickleList):
    with open(pickleFile, 'rb') as file:
        solution = pickle.load(file)
    
    try:
        solution['Primitive']['Density'].shape[1] # indicates multi dimensional primitive arrays: indicating merged results file, processed by the output object, indicating sim successfully finished
        xCoords = solution['X Coords'][1:-1]
        density = solution['Primitive']["Density"][1:-1,-1]
        pressure = solution['Primitive']["Pressure"][1:-1,-1]
        velocity = solution['Primitive']["Velocity"][1:-1,-1]
        mach = solution['Fluid'].computeMach_u_p_rho(velocity, pressure, density)
        entropy = solution['Fluid'].computeEntropy_p_rho(pressure, density)
        totalPressure = solution['Fluid'].computeTotalPressure_p_M(pressure, mach)
        temperature = solution['Fluid'].computeTemperature_p_rho(pressure, density)
        totalTemperature = solution['Fluid'].computeTotalTemperature_T_M(temperature, mach)
    except:
        # only partial finished sim. Solution file arrays are 1D
        xCoords = solution['X Coords'][1:-1]
        density = solution['Primitive']["Density"][1:-1]
        pressure = solution['Primitive']["Pressure"][1:-1]
        velocity = solution['Primitive']["Velocity"][1:-1]
        mach = solution['Fluid'].computeMach_u_p_rho(velocity, pressure, density)
        entropy = solution['Fluid'].computeEntropy_p_rho(pressure, density)
        temperature = solution['Fluid'].computeTemperature_p_rho(pressure, density)
    
    
    axes[0].plot(xCoords, pressure, label=r'$iteration=%s$' %(steplist[i]))
    axes[0].set_ylabel(r'$p$ [Pa]')

    axes[1].plot(xCoords, density)
    axes[1].set_ylabel(r'$\rho$ [kg/m³]')
    
    # axes[1].plot(xCoords, velocity)
    # axes[1].set_ylabel(r'$u$ [m/s]')

    
    
    for ax in axes:
        ax.set_xlabel(r'$x$')
        ax.grid(alpha=.3)

# fig.legend(loc='upper center', bbox_to_anchor=(0.55, 1.18), ncol=3)
fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize = 6)
fig.subplots_adjust(bottom=0.25)
out_root = Path("Pictures") 
out_root.mkdir(parents=True, exist_ok=True)
plt.savefig('Pictures/mach_pressure_ideal_nozzle.pdf', bbox_inches='tight')
plt.show()   
        
        
    