from pyshockflow.fluid import FluidReal

# instantiate FluidReal object for R134a with REFPROP library and abstractstate_v2
fluid = FluidReal("R134a", "REFPROP", "abstractstate_v2")

# inputs to the FluidReal object computeInletQuantitiesTotal method
inputs = {
    "P_stat": 8474000*0.5,  # pressure in Pa
    "P_total": 8474000,  # total pressure in Pa
    "T_total": 313.9,   # total temperature in K
    "direction": 1      # direction of flow
}

density, velocity, energy = fluid.computeInletQuantitiesTotal(inputs["P_stat"], inputs["P_total"], inputs["T_total"], inputs["direction"])

density_v2, velocity_v2, energy_v2 = fluid.computeInletQuantitiesTotal_v2(inputs["P_stat"], inputs["P_total"], inputs["T_total"], inputs["direction"])


print(f"V1 results: Density = {density:.6f} kg/m^3, Velocity = {velocity:.6f} m/s, Energy = {energy:.6f} J/kg")
print(f"V2 results: Density = {density_v2:.6f} kg/m^3, Velocity = {velocity_v2:.6f} m/s, Energy = {energy_v2:.6f} J/kg")
