import base_interface
import numpy as np
from coolprop import coolprop_functions
from CoolProp.CoolProp import PropsSI as CPPropsSI
from CoolProp import AbstractState, PQ_INPUTS
import CoolProp.CoolProp as CP
import re

def remove_between_chars(s, start_char, end_char):
    return re.sub(f'{re.escape(start_char)}.*?{re.escape(end_char)}', '', s)


class CoolPropFluid(base_interface.Fluid):
    # does not return value for rho = 519.20049536, U (internal energy) = 259012.115143168

    def __init__(self, library, name):
        if library == 'CoolProp':
            library = 'HEOS'
        super().__init__(library, name)
        self.get_cmp_cnc()

    def PropsSI(self, prop, x_str, x, y_str, y): 
        
        prop = coolprop_functions.translate_fluidprop_coolprop_prop(prop)
        # print("prop:",prop)
        # print("x_str:", x_str)
        # print("y_str:", y_str)
        str_len = int(len(self.Name))
        if str_len > 3:
            if self.Name[str_len - 3: str_len] == '[1]':
                name = self.Name[0:str_len - 3]
            else:
                name = self.Name
        else:
            name = self.Name
        name = self.Library + '::' + name
        if prop in ['Tcrit', 'Pcrit', 'Tmax', 'M']:
            return CPPropsSI(prop, name)
        elif prop == 'Q':
            out = CPPropsSI(prop, x_str, x, y_str, y, name)
            if out == -1:
                phase = CPPropsSI('Phase', x_str, x, y_str, y, name)
                # when not in VLE zone, coolprop gives -1 as result. Here we make uniform result with Fluidprop
                if phase == 0 or phase == 3:  # corresponds to liquid or liquid above critical pressure
                    return 0
                elif phase == 5 or phase == 1 or phase == 2 or phase == 4:
                    # corresponds to gas, superheated gas, supercritical fluid or critical point
                    return 1
            else:
                return out
        else:
            return CPPropsSI(prop, x_str, x, y_str, y, name)

 



class CoolPropAbstractState:

    def __init__(self, cp_fluid, arguments={}):
        self.Fluid = cp_fluid
        self.FluidPropLanguage = False
        if 'fluidprop_language' in arguments.keys():
            self.FluidPropLanguage = arguments['fluidprop_language']
        self.Is2PhaseHomogeneous = False
        if 'homogeneous_2_phase' in arguments.keys():
            self.Is2PhaseHomogeneous = arguments['homogeneous_2_phase']

        str_len = int(len(cp_fluid.Name))
        if str_len > 3:
            if cp_fluid.Name[str_len - 3: str_len] == '[1]':
                name = cp_fluid.Name[0:str_len - 3]
            else:
                name = cp_fluid.Name
        else:
            name = cp_fluid.Name
        if self.Fluid.nCmp > 1:
            name = remove_between_chars(name, '[', ']')
        self.CPLowLevelInterface = AbstractState(self.Fluid.Library, name)
        if self.Fluid.nCmp > 1:
            self.CPLowLevelInterface.set_mass_fractions(cp_fluid.cnc)
            self.CPLowLevelInterface.build_phase_envelope(cp_fluid.Name)

    def update(self, input_spec, input1, input2):
        self.StashInputSpec = input_spec
        self.StashInput1 = input1
        self.StashInput2 = input2
        if self.FluidPropLanguage:
            input_spec, input1, input2 = (
                coolprop_functions.translate_fluidprop_coolprop_abstractstate_input(input_spec, input1, input2))
        self.CPLowLevelInterface.update(input_spec, input1, input2)

    def p_critical(self):
        return self.CPLowLevelInterface.p_critical()

    def T_critical(self):
        return self.CPLowLevelInterface.T_critical()

    def molar_mass(self):
        return self.CPLowLevelInterface.molar_mass()

    def Tmax(self):
        return self.CPLowLevelInterface.Tmax()

    def p(self):
        return self.CPLowLevelInterface.p()

    def T(self):
        return self.CPLowLevelInterface.T()

    def rhomass(self):
        return self.CPLowLevelInterface.rhomass()

    def hmass(self):
        return self.CPLowLevelInterface.hmass()

    def smass(self):
        return self.CPLowLevelInterface.smass()

    def Q(self):
        phase = self.CPLowLevelInterface.phase()

        if phase == 0 or phase == 3:  # corresponds to liquid or liquid above critical pressure
            return 0
        elif phase == 5 or phase == 1 or phase == 2 or phase == 4:
            # corresponds to gas, superheated gas, supercritical fluid or critical point
            return 1
        else:
            return self.CPLowLevelInterface.Q()

    def cvmass(self):
        if self.Is2PhaseHomogeneous:
            q = self.CPLowLevelInterface.Q()
            if -0.0001 <= q <= 1.0001:
                q = max(0.0, q)
                q = min(1.0, q)
                p = self.CPLowLevelInterface.p()
                self.CPLowLevelInterface.update(PQ_INPUTS, p, 0.0)
                cv1 = self.CPLowLevelInterface.cvmass()
                self.CPLowLevelInterface.update(PQ_INPUTS, p, 1.0)
                cv2 = self.CPLowLevelInterface.cvmass()
                self.CPLowLevelInterface.unspecify_phase()
                self.update(self.StashInputSpec, self.StashInput1, self.StashInput2)
                return q * cv2 + (1 - q) * cv1
        return self.CPLowLevelInterface.cvmass()

    def cpmass(self):
        if self.Is2PhaseHomogeneous:
            q = self.CPLowLevelInterface.Q()
            if -0.0001 <= q <= 1.0001:
                q = max(0.0, q)
                q = min(1.0, q)
                p = self.CPLowLevelInterface.p()
                self.CPLowLevelInterface.update(PQ_INPUTS, p, 0.0)
                cp1 = self.CPLowLevelInterface.cpmass()
                self.CPLowLevelInterface.update(PQ_INPUTS, p, 1.0)
                cp2 = self.CPLowLevelInterface.cpmass()
                self.CPLowLevelInterface.unspecify_phase()
                self.update(self.StashInputSpec, self.StashInput1, self.StashInput2)
                return q * cp2 + (1 - q) * cp1
        return self.CPLowLevelInterface.cpmass()

    def cp0mass(self):
        if self.Is2PhaseHomogeneous:
            q = self.CPLowLevelInterface.Q()
            if -0.0001 <= q <= 1.0001:
                q = max(0.0, q)
                q = min(1.0, q)
                p = self.CPLowLevelInterface.p()
                self.CPLowLevelInterface.update(PQ_INPUTS, p, 0.0)
                cp1 = self.CPLowLevelInterface.cp0mass()
                self.CPLowLevelInterface.update(PQ_INPUTS, p, 1.0)
                cp2 = self.CPLowLevelInterface.cp0mass()
                self.CPLowLevelInterface.unspecify_phase()
                self.update(self.StashInputSpec, self.StashInput1, self.StashInput2)
                return q * cp2 + (1 - q) * cp1
        return self.CPLowLevelInterface.cp0mass()

    def cp0molar(self):
        if self.Is2PhaseHomogeneous:
            q = self.CPLowLevelInterface.Q()
            if -0.0001 <= q <= 1.0001:
                q = max(0.0, q)
                q = min(1.0, q)
                p = self.CPLowLevelInterface.p()
                self.CPLowLevelInterface.update(PQ_INPUTS, p, 0.0)
                cp1 = self.CPLowLevelInterface.cp0molar()
                self.CPLowLevelInterface.update(PQ_INPUTS, p, 1.0)
                cp2 = self.CPLowLevelInterface.cp0molar()
                self.CPLowLevelInterface.unspecify_phase()
                self.update(self.StashInputSpec, self.StashInput1, self.StashInput2)
                return q * cp2 + (1 - q) * cp1
        return self.CPLowLevelInterface.cp0molar()

    def speed_sound(self):

        if self.Is2PhaseHomogeneous:
            q = self.CPLowLevelInterface.Q()
            if -0.0001 <= q <= 1.0001:
                q = max(0.0,q)
                q = min(1.0,q)
                p = self.CPLowLevelInterface.p()
                self.CPLowLevelInterface.update(PQ_INPUTS, p, 0.0)
                ss1 = self.CPLowLevelInterface.speed_sound()
                self.CPLowLevelInterface.update(PQ_INPUTS, p, 1.0)
                ss2 = self.CPLowLevelInterface.speed_sound()
                self.CPLowLevelInterface.unspecify_phase()
                self.update(self.StashInputSpec, self.StashInput1, self.StashInput2)
                return q*ss2 + (1-q)*ss1
        return self.CPLowLevelInterface.speed_sound()

    def fundamental_derivative_of_gas_dynamics(self):
        if self.Is2PhaseHomogeneous:
            q = self.CPLowLevelInterface.Q()
            if -0.0001 <= q <= 1.0001:
                q = min(1.0, q)
                p = self.CPLowLevelInterface.p()
                self.CPLowLevelInterface.update(PQ_INPUTS, p, 0.0)
                fdgd1 = self.CPLowLevelInterface.fundamental_derivative_of_gas_dynamics()
                self.CPLowLevelInterface.update(PQ_INPUTS, p, 1.0)
                fdgd2 = self.CPLowLevelInterface.fundamental_derivative_of_gas_dynamics()
                self.CPLowLevelInterface.unspecify_phase()
                self.update(self.StashInputSpec, self.StashInput1, self.StashInput2)
                return q * fdgd2 + (1 - q) * fdgd1

        return self.CPLowLevelInterface.fundamental_derivative_of_gas_dynamics()

    def viscosity(self):
        return self.CPLowLevelInterface.viscosity()

    def conductivity(self):
        return self.CPLowLevelInterface.conductivity()

    def compressibility_factor(self):
        return self.CPLowLevelInterface.compressibility_factor()

    def drhomassdPcT(self):
        # turbosim compatibility
        of = coolprop_functions.CoolProp.iDmass
        wrt = coolprop_functions.CoolProp.iP
        const = coolprop_functions.CoolProp.iT
        return self.CPLowLevelInterface.first_partial_deriv(of, wrt, const)

    def first_partial_deriv(self, of, wrt, const):
        # turbosim compatibility
        if of == 'rhomass' and wrt == 'P' and const == 'T':
            of = coolprop_functions.CoolProp.iDmass
            wrt = coolprop_functions.CoolProp.iP
            const = coolprop_functions.CoolProp.iT

        return self.CPLowLevelInterface.first_partial_deriv(of, wrt, const)






def update_wrapper(AS, input_spec, x, y):
        """
        Coolprop utility to allow nan return upon vectorized evaluation of AbstractState.
        """
        try:
            AS.update(input_spec, x, y)
            return False
        except ValueError:
            return True

class CoolPropAbstractState_v2():
    def __init__(self, library, name):
        if library == 'CoolProp':
            library = 'HEOS'
        self.Name = name
        self.Library = library

    def PropsSI_syntax_to_AbstractState_syntax(self, str):
        mass_props_list = ["D", "U", "H", "S"]
        if str in mass_props_list:
            return str + "mass"
        else:
            return str
        
    def get_input_spec(self, x_str, y_str):
        supported_input_specs = [
            "PT", "PUmass", "DmassP", "HmassP", "PQ", "DmassT", "DmassUmass", 
            "DmassHmass", "DmassQ", "TUmass", "HmassT", "QT", "SmassT", "SmassUmass", 
            "DmassSmass", "HmassSmass", "QSmass", "PSmass"
            ]
        if x_str + y_str in supported_input_specs:
            reorder = False
            return getattr(CP, x_str + y_str + "_INPUTS"), reorder
        else:
            reorder = True
            return getattr(CP, y_str + x_str + "_INPUTS"), reorder
        
    @np.vectorize(otypes=[float]) # update abstract state can only happen with single thermodynamic point, arrays are not supported.
    def update_and_get(AS, input_spec, x, y, output, reorder): # need no arg self since the vectorize decorator adds this automatically
        if reorder:
            skip_update = update_wrapper(AS, input_spec, y, x)
        else:
            skip_update = update_wrapper(AS, input_spec, x, y)
        if skip_update:
            return np.nan
        if output == 'drhomassdPcT':
            return AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iT)
        return getattr(AS, output)()    
    
    def PropsSI(self, prop, x_str=None, x=None, y_str=None, y=None):
        str_len = int(len(self.Name))
        if str_len > 3:
            if self.Name[str_len - 3: str_len] == '[1]':
                name = self.Name[0:str_len - 3]
            else:
                name = self.Name
        else:
            name = self.Name
        AS = AbstractState(self.Library, name)
        if prop in ['Tcrit', 'Pcrit', 'Dcrit', 'Tmax', 'M', 'Ttriple']: # translation necessary to comply with AS syntax, see: https://coolprop.org/_static/doxygen/html/class_cool_prop_1_1_abstract_state.html
            if prop== 'Tcrit':
                return AS.T_critical()
            elif prop == 'Pcrit':
                return AS.p_critical()
            elif prop == 'Dcrit':
                return AS.rhomass_critical()    
            elif prop == 'Tmax':
                return AS.Tmax()
            elif prop == 'M':
                return AS.molar_mass()
            elif prop == 'Ttriple':
                return AS.Ttriple()
        prop_AS = self.PropsSI_syntax_to_AbstractState_syntax(prop)
        x_str_AS = self.PropsSI_syntax_to_AbstractState_syntax(x_str)
        y_str_AS = self.PropsSI_syntax_to_AbstractState_syntax(y_str)
        input_spec, reorder = self.get_input_spec(x_str_AS, y_str_AS)
        if prop_AS == 'Q':
            out = self.update_and_get(AS, input_spec, x, y, prop_AS, reorder)
            if out == -1:
                phase = self.update_and_get(AS, input_spec, x, y, 'phase', reorder)
                # when not in VLE zone, coolprop gives -1 as result. Here we make uniform result with Fluidprop
                if phase == 0 or phase == 3:  # corresponds to liquid or liquid above critical pressure
                    return 0
                elif phase == 5 or phase == 1 or phase == 2 or phase == 4:
                    # corresponds to gas, superheated gas, supercritical fluid or critical point
                    return 1
            else:
                return out
        else:
            translator = {
                "Umass": "umass",
                "Dmass": "rhomass",
                "Hmass": "hmass",
                "A": "speed_sound",
                "T": "T",
                "Q": "Q",
                "P": "p",
                "Smass": "smass",
                "Cpmass": "cpmass",
                "Cvmass": "cvmass",
                "d(P)/d(D)|T": "drhomassdPcT",
            }
            return self.update_and_get(AS, input_spec, x, y, translator[prop_AS], reorder)
