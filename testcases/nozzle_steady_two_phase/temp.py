
from fluid_properties.coolprop_interface import CoolPropAbstractState_v2
from pyshockflow.fluid import FluidReal

AS = CoolPropAbstractState_v2("REFPROP", "R1234ze(E)")
FluidRealObj = FluidReal("R1234ze(E)", "REFPROP", "abstractstate_v2")

p = AS.PropsSI("P", "S", 1600.72573613, "T", 382.34715144)  
rho = AS.PropsSI("D", "S", 1600.72573613, "T", 382.34715144)
print("p: ", p)
print("rho: ", rho)

SOS = FluidRealObj.computeSoundSpeed_p_rho(p, rho)
print(SOS)

print(FluidRealObj.computeSoundSpeed_p_rho(3622983.93194916, 409.35743802))

print(FluidRealObj.computeSoundSpeed_p_rho(3622983.931949161, 409.3574380247973))

print(FluidRealObj.computeSoundSpeed_p_rho(3622983.931949161, 409.3574380247973))