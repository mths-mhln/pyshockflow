import CoolProp
import base_interface
from coolprop import ha_coolprop_functions
from CoolProp.HumidAirProp import HAPropsSI
from CoolProp.CoolProp import PropsSI
import numpy as np

from scipy.optimize import brentq, least_squares

class HumidAirMixture(base_interface.Fluid):

    def __init__(self, composition = 'sea_level_2_comp'):


        components = self.setAirComposition(composition)

        self._abshum_string = ['abshum','W']
        self._relhum_string = ['relhum','R']

        self.cmp = components
        self.cmp.append('water')
        self.nCmp = len(self.cmp)

        super().__init__('Humid Air Mix', '&'.join(cmp for cmp in self.cmp))

        self.setAbstractState()

        try:

            self.absolute_humidity

            self.setHumidity('W', self.absolute_humidity)

        except:

            pass

    def setAirComposition(self, spec):

        # convert mole fractions to mass fractions

        if spec == 'sea_level_2_comp' or spec == 'sea_level_4_comp' or (type(spec) is dict and 'molar_composition' in spec.keys()):

            if spec == 'sea_level_2_comp':

                components = ['nitrogen', 'oxygen']
                mole_fractions = [0.79, 0.21]

            elif spec == 'sea_level_4_comp':

                components = ['nitrogen', 'oxygen', 'argon', 'co2']
                mole_fractions = [0.7808, 0.2095, 0.0093, 0.0004]

            else:

                spec_type = 'molar_composition'

                components = list(spec[spec_type].keys())
                mole_fractions = list(spec[spec_type].values())

            as_air_tmp = CoolProp.AbstractState('REFPROP', '&'.join(cmp for cmp in components))
            as_air_tmp.set_mole_fractions(mole_fractions)
            mass_fractions = as_air_tmp.get_mass_fractions()


        elif type(spec) is dict and 'mass_composition' in spec.keys():

                spec_type = 'mass_composition'

                components = list(spec[spec_type].keys())
                mass_fractions = list(spec[spec_type].values())

        else:

            raise Exception ('Unrecognized composition specified.')


        if 'water' not in components:

            self._air_components_mass_fractions_per_kg_air = mass_fractions

        else:

            loc = components.index('water')
            water_mass_fraction = mass_fractions[loc]
            self._air_components_mass_fractions_per_kg_air = [mass / (1 - water_mass_fraction) for i,mass in enumerate(mass_fractions) if i != loc]
            self._air_mass_fraction = 1 - water_mass_fraction

            self.absolute_humidity = water_mass_fraction / self._air_mass_fraction

            components.remove('water')

        return components


    def setAbstractState(self):

        self.AbstractState = CoolProp.AbstractState('REFPROP', self.Name)

    def setHumidity(self, prop, val):

        if prop in self._abshum_string:

            self._air_mass_fraction_per_kg_humidair = 1 / (1 + val)

            self._air_components_mass_fractions_per_kg_humidair = np.array(self._air_components_mass_fractions_per_kg_air) * self._air_mass_fraction_per_kg_humidair

            self._humidair_components_mass_fractions_per_kg_humidair = list(self._air_components_mass_fractions_per_kg_humidair)
            self._humidair_components_mass_fractions_per_kg_humidair.append(val * self._air_mass_fraction_per_kg_humidair)

            self.AbstractState.set_mass_fractions(self._humidair_components_mass_fractions_per_kg_humidair)

            self.absolute_humidity = val

        else:

            raise NotImplementedError ('Feature not implemented yet.')


class HumidAirAbstractState:

    def __init__(self, humid_air_fluid):
        self.Fluid = humid_air_fluid

        self._abshum_string = ['abshum','W']

    def set_humidity(self, prop, val):

        if prop in self._abshum_string:

            self.Fluid.setHumidity(prop, val)

        else:
            raise NotImplementedError (f'Feature not implemented yet. Available options are: prop = {self._abshum_string} '
                                       f'for setting the absolute humidity directly, or prop = {self._relhum_string} '
                                       f'for setting the relative humidity.')

    def update(self, InputSpec, Input1, Input2):
        self.InputSpec1, self.InputSpec2 = ha_coolprop_functions.translate_coolprop_abstractstate_inputspec(InputSpec)
        self.Input1 = Input1
        self.Input2 = Input2
        self.Fluid.AbstractState.update(InputSpec, Input1, Input2)

    def p_critical(self):
        return self.Fluid.AbstractState.p_critical()

    def T_critical(self):
        return self.Fluid.AbstractState.T_critical()

    def molar_mass(self):
        return self.Fluid.AbstractState.molar_mass()

    def p(self):
        return self.Fluid.AbstractState.p()

    def T(self):
        return self.Fluid.AbstractState.T()

    def rhomass(self):
        return self.Fluid.AbstractState.rhomass()

    def hmass(self):
        return self.Fluid.AbstractState.hmass()

    def smass(self):
        return self.Fluid.AbstractState.smass()

    def Q(self):
        raise NotImplementedError ('Property call not implemented yet.')

    def cvmass(self):
        return self.Fluid.AbstractState.cpmass() - (self.gas_constant() / self.molar_mass())

    def cpmass(self):
        return self.Fluid.AbstractState.cpmass()

    def cv0mass(self):
        return self.Fluid.AbstractState.cp0mass() - (self.gas_constant() / self.molar_mass())

    def cp0mass(self):
        return self.Fluid.AbstractState.cp0mass()

    def speed_sound(self):
        return self.Fluid.AbstractState.speed_sound()

    def fundamental_derivative_of_gas_dynamics(self):
        return self.Fluid.AbstractState.fundamental_derivative_of_gas_dynamics()

    def viscosity(self):
        return self.Fluid.AbstractState.viscosity()

    def conductivity(self):
        return self.Fluid.AbstractState.conductivity()

    def compressibility_factor(self):
        return self.Fluid.AbstractState.compressibility_factor()

    def gamma_pv(self):
        dP_dv_T = (- 1 / (self.rhomass() ** 2) *
                   self.Fluid.AbstractState.first_partial_deriv(CoolProp.iDmass, CoolProp.iP, CoolProp.iT)) ** (-1)
        return - 1 / (self.p() * self.rhomass()) * self.cpmass() / self.cvmass() * dP_dv_T

    def gamma(self):
        return self.cp0mass() / self.cv0mass()

    def get_mass_fractions(self):
        return self.Fluid.AbstractState.get_mass_fractions()

    def absolute_humidity(self):
        return self.Fluid.absolute_humidity

    def gas_constant(self):
        return self.Fluid.AbstractState.gas_constant()

class HumidAirCoolPropFluid(base_interface.Fluid):

    def __init__(self):
        super().__init__('Humid Air', 'Humid Air')
        self.cmp = ['air', 'water']
        self.nCmp = 2

        self.skip_next = False

    def all_prop(self):

        self.Tcrit = 132.5306
        self.Pcrit = 3786000.0
        self.Mmol = 0.02896546
        self.sigma = -8.264545220895442
        self.sigma1 = 0

    def set_humid_air_absolute_humidity(self, T, P, R):
        self.AbsoluteHumidity = ha_coolprop_functions.get_absolute_humidity(T, P, R)

    def PropsSI(self, prop, x_str, x, y_str, y):

        if x_str == 'Q' or y_str == 'Q':
            return PropsSI(prop, x_str, x, y_str, y, 'air')

        elif prop == 'Q':
            return 1.0

        elif prop == 'RH':

            return HAPropsSI(prop, x_str, x, y_str, y, 'W', self.AbsoluteHumidity)

        else:
            prop_new = ha_coolprop_functions.translate_prop(prop)

            if 'P' in [x_str, y_str]:

                return 1 / HAPropsSI(prop_new, x_str, x, y_str, y, 'W', self.AbsoluteHumidity) if prop == 'D' \
                    else HAPropsSI(prop_new, x_str, x, y_str, y, 'W', self.AbsoluteHumidity)

            else:

                if not self.skip_next:

                    try:

                        self.P = brentq(self.PressureLoop, 0.1e5, 3e5, args=(x_str, x, y_str, y))

                    except:

                        self.P = least_squares(self.PressureLoop, 1e5, args=(x_str, x, y_str, y)).get('x')[0]

                    self.skip_next = True

                return 1 / HAPropsSI(prop_new, 'P', self.P, y_str, y, 'W', self.AbsoluteHumidity) if prop == 'D' \
                    else self.P if prop == 'P' else HAPropsSI(prop_new, 'P', self.P, y_str, y, 'W', self.AbsoluteHumidity)

    def PressureLoop(self, p, x_str, x, y_str, y):

        P = p

        try:

            if x_str == 'D':

                x_new = 1 / HAPropsSI('Vha', 'P', P, y_str, y, 'W', self.AbsoluteHumidity)

            else:

                x_new = HAPropsSI(x_str, 'P', P, y_str, y, 'W', self.AbsoluteHumidity)



        except:
            x_new = np.inf

        res = x_new - x

        return res



class HumidAirCoolPropAbstractState:

    def __init__(self, humid_air_fluid):
        self.Fluid = humid_air_fluid

        self._abshum_string = ['abshum','W']
        self._relhum_string = ['relhum','R']

    def update(self, InputSpec, Input1, Input2):
        self.InputSpec1, self.InputSpec2 = ha_coolprop_functions.translate_coolprop_abstractstate_inputspec(InputSpec)
        self.Input1 = Input1
        self.Input2 = Input2

        if 'P' not in [self.InputSpec1, self.InputSpec2]:
            self.Fluid.skip_next = False

    def set_humidity(self, prop, val, T=None, P=None):

        if prop in self._abshum_string:

            self.Fluid.AbsoluteHumidity = val

        elif prop in self._relhum_string:

            if T is not None and P is not None: self.Fluid.set_humid_air_absolute_humidity(T, P, val)
            else: raise Exception ('When setting the relative humidity, dry bulb temperature (T) and pressure (P) must '
                                   'be specified as inputs')

        else:
            raise NotImplementedError (f'Feature not implemented yet. Available options are: prop = {self._abshum_string} '
                                       f'for setting the absolute humidity directly, or prop = {self._relhum_string} '
                                       f'for setting the relative humidity.')

    def p_critical(self):
        return self.Fluid.Pcrit

    def T_critical(self):
        return self.Fluid.Tcrit

    def molar_mass(self):
        return self.Fluid.Mmol

    def p(self):
        return self.Fluid.PropsSI('P', self.InputSpec1, self.Input1, self.InputSpec2, self.Input2)

    def T(self):
        return self.Fluid.PropsSI('T', self.InputSpec1, self.Input1, self.InputSpec2, self.Input2)

    def rhomass(self):
        return self.Fluid.PropsSI('D', self.InputSpec1, self.Input1, self.InputSpec2, self.Input2)

    def hmass(self):
        return self.Fluid.PropsSI('H', self.InputSpec1, self.Input1, self.InputSpec2, self.Input2)

    def smass(self):
        return self.Fluid.PropsSI('S', self.InputSpec1, self.Input1, self.InputSpec2, self.Input2)

    def Q(self):
        return 1.0

    def cvmass(self):
        return self.Fluid.PropsSI('CV', self.InputSpec1, self.Input1, self.InputSpec2, self.Input2)

    def cpmass(self):
        return self.Fluid.PropsSI('C', self.InputSpec1, self.Input1, self.InputSpec2, self.Input2)

    def cv0mass(self):
        return self.Fluid.PropsSI('CV', 'P', 1e5, 'T', 500)

    def cp0mass(self):
        return self.Fluid.PropsSI('C', 'P', 1e5, 'T', 500)

    def speed_sound(self):
        return np.sqrt(self.cpmass() / self.cvmass() * self.p() / self.rhomass())

    def fundamental_derivative_of_gas_dynamics(self):
        return (self.cpmass() / self.cvmass() + 1) / 2

    def viscosity(self):
        return self.Fluid.PropsSI('V', self.InputSpec1, self.Input1, self.InputSpec2, self.Input2)

    def conductivity(self):
        return self.Fluid.PropsSI('K', self.InputSpec1, self.Input1, self.InputSpec2, self.Input2)

    def compressibility_factor(self):
        return self.p() / (self.rhomass() * 8.314 / self.molar_mass() * self.T())

    def gamma_pv(self):
        return self.cpmass() / self.cvmass()

    def relative_humidity(self):
        return self.Fluid.PropsSI('RH', self.InputSpec1, self.Input1, self.InputSpec2, self.Input2)

    def absolute_humidity(self):
        return self.Fluid.AbsoluteHumidity

    def dew_point(self):
        return self.Fluid.PropsSI('DewPoint', self.InputSpec1, self.Input1, self.InputSpec2, self.Input2)

if __name__ == '__main__':

    # Composition = \
    #     {
    #         'mass_composition':
    #             {
    #                 'nitrogen':     0.638,
    #                 'oxygen':       0.162,
    #                 'water':        0.2,
    #             },
    #     }
    #
    # p = 1e5
    # T = 300
    # war = 0.0
    #
    # eos = HumidAirAbstractState(HumidAirMixture(composition = 'sea_level_4_comp'))
    # eos.set_humidity('W',war)
    # eos.update(CoolProp.PT_INPUTS, p, T)

    eos_cp = HumidAirCoolPropAbstractState(HumidAirCoolPropFluid())
    eos_cp.Fluid.all_prop()
    eos_cp.Fluid.AbsoluteHumidity = 0.07
    eos_cp.update(CoolProp.PSmass_INPUTS, 39000, 558.5984909789124)
    print(eos_cp.rhomass())
    print(eos_cp.T())