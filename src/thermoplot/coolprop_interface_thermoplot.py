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
    """
    CoolProp AbstractState wrapper. allows user to use the familiar PropsSI syntax for CoolProp property extraction, while using the AbstractState under the hood for better performance. 
    The wrapper is necessary to allow vectorized evaluation of the AbstractState, which is not natively supported by CoolProp. Nan will be returned for points that are not valid for the 
    AbstractState (e.g. points outside the phase envelope). For more information on the AbstractState and its methods, see: https://coolprop.org/_static/doxygen/html/class_cool_prop_1_1_abstract_state.html
    
    Methods
    -------
    PropsSI(prop, x_str, x, y_str, y)
        Extracts the specified property using the AbstractState. The input specification is automatically determined based on the x_str and y_str arguments, and the property is extracted using the 
        appropriate AbstractState method. For more information on the input specifications, see: https://coolprop.org/coolprop/wrappers/Python/html/index.html#input-specifications    
    """

    def __init__(self, library, name):
        """
        Initializes the CoolPropAbstractState object with the specified library and fluid name. The library is typically "HEOS" for pure fluids, but can be adapted for mixtures and other libraries. 
        The name is the name of the fluid as recognized by CoolProp, e.g. "Water" or "R134a". 

        Attributes
        ----------
        Library: str
            Name of the backend library to use for extracting fluid thermodynamic properties
        Name: str
            Name of the fluid as recognized by CoolProp.
        """
        if library == 'CoolProp':
            library = 'HEOS'
        self.Name = name
        self.Library = library

    def PropsSI_syntax_to_AbstractState_syntax(self, str):
        """
        Converts PropsSI syntax to AbstractState syntax. for properties that are typically mass-averaged, the subscript mass should be added behind it.
        """
        mass_props_list = ["D", "U", "H", "S"]
        if str in mass_props_list:
            return str + "mass"
        else:
            return str
        
    def get_input_spec(self, x_str, y_str):
        """
        Method to convert specified PropsSI input spec into a coolprop inputs object, required for updating the abstractstate thermodynamic state in update_and_get using the coolprop abstractstate
        update method. 

        Attributes
        ----------
        x_str: str
            String corresponding to the first input variable, e.g. "T" for temperature or "P" for pressure.
        y_str: str
            String corresponding to the second input variable, e.g. "T" for temperature or "P" for pressure.
        """
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
        """
        Vectorized method to update the AbstractState with the specified input specification and input variables, and return the specified output variable. 
        The method returns nan for points that are not valid for the AbstractState (e.g. points outside the phase envelope).

        Arguments
        ---------
        AS: AbstractState
            CoolProp AbstractState object to update and extract properties from.
        input_spec: int
            CoolProp input specification corresponding to the x_str and y_str variables, e.g. CP.PT_INPUTS for temperature and pressure inputs. This is determined in the get_input_spec method.
        x: float
            Value of the first input variable, e.g. temperature or pressure.
        y: float
            Value of the second input variable, e.g. temperature or pressure.
        output: str
            String corresponding to the desired output variable, e.g. "T" for temperature or "P" for pressure. This is translated to the corresponding AbstractState method in the PropsSI method.
        reorder: bool
            Boolean indicating whether the input variables need to be reordered for the AbstractState update method as a specific order may be required to comply with CoolProp AbstractState syntax.
        
        Returns
        -------
        output: float | np.ndarray
            output of the desired variable, e.g. temperature or pressure. Will be a float for single point evaluation, or a numpy array for vectorized evaluation. 
            For points that are not valid for the AbstractState (e.g. points outside the phase envelope), nan will be returned.
        """
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
        """
        Integral functionality, uses various methods to convert user input to an input spec accepted by AbstractState syntax, and extracts fluid thermodynamic property according to user specification. 
        using a CoolProp AbstractState syntax. 

        Attributes
        ----------
        prop: str
            String corresponding to the desired output variable, e.g. "T" for temperature or "P" for pressure. 
        x_str: str
            String corresponding to one of the input variables, e.g. "T" for temperature or "P" for pressure.
        x: float | np.ndarray
            Value of the first input variable, e.g. temperature or pressure. Can be a float for single point evaluation, or a numpy array for vectorized evaluation.
        y_str: str
            String corresponding to the other input variable, e.g. "T" for temperature or "P" for pressure.
        y: float | np.ndarray
            Value of the second input variable e.g. temperature or pressure. Can be a float for single point evaluation, or a numpy array for vectorized evaluation.
        
        Result
        ------
        output: float | np.ndarray
            Value of the desired output variable, e.g. temperature or pressure. Will be a float for single point evaluation, or a numpy array for vectorized evaluation. 
            For points that are not valid for the AbstractState (e.g. points outside the phase envelope), nan will be returned.       
        """
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
        



