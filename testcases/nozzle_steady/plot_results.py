import numpy as np
import matplotlib.pyplot as plt
import pickle

from pathlib import Path
from pyshockflow.plot_styles import *
from scipy.optimize import fsolve

# Extract all pickle files from Results folder
pickleList = sorted(Path("Results/outletPressure_45kPa_NX_200").glob("*.pik"))

# extract step digit from pickle file names, to use as legend labels and remove leading 00
stepList = [pickleFile.stem.split("_")[-1].lstrip('0') for pickleFile in pickleList]

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for i, pickleFile in enumerate(pickleList):
    with open(pickleFile, 'rb') as file:
        solution = pickle.load(file)
    config = solution['Configuration']
    
    if config.getFluidModel() == "ideal" and i==0: # compute also reference from nozzle theory
        xArea = solution['X Coords']
        areaRatio =  solution['Area Tube'] / np.min(solution['Area Tube'])
        gammaFluid = solution['Fluid'].gmma
        def machFunction(machLocal, areaRatioLocal, gammaFluid):
            residual = areaRatioLocal - 1/machLocal * (2/(gammaFluid+1) * (1 + (gammaFluid-1)/2 * machLocal**2))**((gammaFluid+1)/(2*(gammaFluid-1)))
            return residual

        machTheory = np.zeros(len(xArea))
        idThroat = np.argmin(areaRatio)
        for iPoint in range(len(xArea)):
            if iPoint < idThroat:
                machTheory[iPoint] = fsolve(machFunction, 0.1, args=(areaRatio[iPoint], gammaFluid))[0]
            else:
                machTheory[iPoint] = fsolve(machFunction, 1.2, args=(areaRatio[iPoint], gammaFluid))[0]

        axes[0].plot(xArea[::10], machTheory[::10], 'ko', mfc='none' ,label=r'Supersonic reference')
    
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
    if i==0:
        print(pressure)
    axes[0].plot(xCoords, pressure, label=r'$iteration=%s$' %(stepList[i]))
    axes[0].set_ylabel(r'$p$ [Pa]')
    
    axes[1].plot(xCoords, density)
    axes[1].set_ylabel(r'$\rho$ [kg/m$^3$]')
    
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
    
        
        
        
    