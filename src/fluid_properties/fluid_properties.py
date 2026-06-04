from fluidprop_interface import FluidPropFluid, FluidPropAbstractState
from coolprop_interface import CoolPropFluid, CoolPropAbstractState, CoolPropAbstractState_v2
from coolprop_humidair_interface import HumidAirCoolPropFluid, HumidAirCoolPropAbstractState, HumidAirMixture, HumidAirAbstractState
from feos_interface import FeosFluid, FeosAbstractState
from lut_interface import LuTFluid, LuTAbstractState

FluidPropLibraries = ['StanMix', 'GasMix', 'PCP-SAFT', 'RefProp', 'qPCP-SAFT', 'HOGC-PCP-SAFT']
CoolPropLibraries = ['CoolProp', 'REFPROP', 'HEOS']
HumidAirLibraries = ['Humid Air', 'Humid Air Mix']
LuTLibraries = ['LuT']
FeosLibraries = ['feos::HOGC-PCP-SAFT']


def fluid(library, name, **kwargs):
    if library in FluidPropLibraries:
        fluid_object = FluidPropFluid(library, name, kwargs)
    elif library in CoolPropLibraries:
        fluid_object = CoolPropFluid(library, name)
    elif library in HumidAirLibraries:
        fluid_object = HumidAirMixture() if library == 'Humid Air Mix' else HumidAirCoolPropFluid()
    elif library in FeosLibraries:
        fluid_object = FeosFluid(library, name, kwargs)
    else:
        raise Exception('library not recognized')

    if 'use_lut' in kwargs.keys():
        if kwargs['use_lut']:
            fluid_object = build_lookup_table_fluid(fluid_object, kwargs)
    return fluid_object


def AbstractState(fluid_object, **kwargs):
    if fluid_object.Library in FluidPropLibraries:
        return FluidPropAbstractState(fluid_object, kwargs)
    elif fluid_object.Library in CoolPropLibraries:
        return CoolPropAbstractState(fluid_object, kwargs)
    elif fluid_object.Library in HumidAirLibraries:
        return HumidAirAbstractState(fluid_object) if fluid_object.Library == 'Humid Air Mix' else HumidAirCoolPropAbstractState(fluid_object)
    elif fluid_object.Library in FeosLibraries:
        return FeosAbstractState(fluid_object, kwargs)
    elif fluid_object.Library in LuTLibraries:
        return LuTAbstractState(fluid_object, kwargs)
    
def AbstractState_v2(library, name, **kwargs):
    if library in CoolPropLibraries:
        fluid_object = CoolPropAbstractState_v2(library, name)
        return fluid_object

def add_lookup_table(lut_fluid, fluid_object, **kwargs):
    kwargs['fluidprop_language'] = True
    abstractstate = AbstractState(fluid_object, **kwargs)
    lut_fluid.add_lut(abstractstate, kwargs)


def build_lookup_table_fluid(fluid_object, arguments):
    arguments['fluidprop_language'] = True
    abstractstate = AbstractState(fluid_object, **arguments)
    return LuTFluid(abstractstate, arguments)


def PropsSI(prop, x_str, x, y_str, y, fluid_object):
    return fluid_object.PropsSI(prop, x_str, x, y_str, y)

if __name__ == '__main__':
    import CoolProp
    FluidBase = fluid('Humid Air Mix','nitrogen&oxygen&water')

    Fluid = fluid('Humid Air Mix','nitrogen&oxygen&water',
                  use_lut = True,
                  import_lut=True,lut_data_file_name = 'lookup_tables/nitrogen&oxygen&water/PTBounds.json',
                  node_values_file_name = 'lookup_tables/nitrogen&oxygen&water/PTGrids.npy',
                  fallback_to_eos = True,
                  low_level_library = 'REFPROP')
    add_lookup_table(Fluid, FluidBase,
                     import_lut=True,lut_data_file_name = 'lookup_tables/nitrogen&oxygen&water/PHBounds.json',
                     node_values_file_name = 'lookup_tables/nitrogen&oxygen&water/PHGrids.npy',
                     fallback_to_eos = True,
                     low_level_library = 'REFPROP')
    add_lookup_table(Fluid, FluidBase,
                     import_lut=True,lut_data_file_name = 'lookup_tables/nitrogen&oxygen&water/PSBounds.json',
                     node_values_file_name = 'lookup_tables/nitrogen&oxygen&water/PSGrids.npy',
                     fallback_to_eos = True,
                     low_level_library = 'REFPROP')
    add_lookup_table(Fluid, FluidBase,
                     import_lut=True,lut_data_file_name = 'lookup_tables/nitrogen&oxygen&water/HSBounds.json',
                     node_values_file_name = 'lookup_tables/nitrogen&oxygen&water/HSGrids.npy',
                     fallback_to_eos = True,
                     low_level_library = 'REFPROP')

    eos = AbstractState(Fluid)
    eos.update('PT', 1e5, 300)
    print(eos.rhomass())
    eos.update('hs', eos.hmass(), eos.smass())
    print(eos.rhomass())

    eos = AbstractState(FluidBase)
    eos.set_humidity('W',0.01)
    eos.update(CoolProp.PT_INPUTS, 1e5, 300)
    print(eos.rhomass())