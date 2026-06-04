Simulations used to show steady-state nozzle solution capabilities for ideal gas.
A nozzle with a quadratic shape is defined in write_nozzle.py, and the area distribution is used
in the simulation main.py where the nozzle is the whole domain. 
For this purpose inlet/outlet boundary conditions are used.
For inlet values 1 bar and 288.15K are used as total pressure and temperature, while the
static pressure at outlet is varied from subsonic to supersonic values. For intermediate situations, a
shock location in the divergent part of the nozzle is visible.