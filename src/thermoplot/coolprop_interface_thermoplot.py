import numpy as np
from CoolProp import AbstractState
import CoolProp.CoolProp as CP



# Utilities
def update_wrapper(AS, input_spec, x, y):
        """
        Coolprop utility to allow nan return upon vectorized evaluation of AbstractState.
        """
        try:
            AS.update(input_spec, x, y)
            return False
        except ValueError:
            return True

class CoolPropAbstractState():
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
        



