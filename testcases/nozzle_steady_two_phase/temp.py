from fluid_properties.coolprop_interface import CoolPropAbstractState_v2

AS = CoolPropAbstractState_v2("REFPROP", "R134a")

print(AS.PropsSI("T", "P", 101325, "S", 2000))  # Saturation temperature at 1 atm

