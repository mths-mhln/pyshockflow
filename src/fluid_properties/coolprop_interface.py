from functools import lru_cache

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
            if type(out) == float:  # scalar output
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
            else:  # array output
                out[out < 0] = 0
                out[out > 1] = 1
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



import numpy as np
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState


class CoolPropAbstractState_v2():
    """
    CoolProp AbstractState wrapper. allows user to use the familiar PropsSI syntax for CoolProp property extraction, while using the AbstractState under the hood for better performance. 
    The wrapper is necessary to allow vectorized evaluation of the AbstractState, which is not natively supported by CoolProp. Nan will be returned for points that are not valid for the 
    AbstractState (e.g. points outside the phase envelope). For more information on the AbstractState and its methods, see: https://coolprop.org/_static/doxygen/html/class_cool_prop_1_1_abstract_state.html
    
    Methods
    -------
    PropsSI(prop, x_str, x, y_str, y)
        Extracts the specified property using the AbstractState. The input specification is automatically determined based on the x_str and y_str arguments, and the property is extracted using the 
        appropriate AbstractState method. For more information on the input specifications, see: https://coolprop.org/coolprop/wrappers/Python/html/index.html#input-specifications

        `prop` may also be a tuple of property strings, e.g. ("T", "P", "Dmass"). In that case, the AbstractState is
        updated only ONCE per (x, y) point, and all requested properties are extracted from that single update. This
        avoids the cost of repeatedly calling AS.update() for every property you want at the same state point. The
        return value is then a tuple of floats/arrays (one per requested property, in the order given), matching the
        order of `prop`.
    """
    # ------------------------------------------------------------------
    # Lookup Tables. Replace old computational routines by a simple lookup, 
    # which allowed for less computations. 
    # ------------------------------------------------------------------
    _MASS_PROPS = frozenset({"D", "U", "H", "S"})

    # Translation from CoolProp PropsSI method syntax to syntax 
    # accepted by CoolProp Abstractstate. See
    # https://coolprop.org/_static/doxygen/html/class_cool_prop_1_1_abstract_state.html
    _TRANSLATOR = {
        "Umass": "umass",
        "Dmass": "rhomass",
        "Hmass": "hmass",
        "A": "speed_sound",
        "T": "T",
        "Q": "Q",
        "P": "p",
        "G": "gibbsmass",
        "Smass": "smass",
        "Cpmass": "cpmass",
        "Cvmass": "cvmass",
        "d(P)/d(D)|T": "drhomassdPcT",          
        "d(P)/d(D)|U": "dPdDmass_constUmass",   
        "d(P)/d(U)|D": "dPdUmass_constDmass",  
        "Phase": "phase",
        "V": "viscosity",
    }

    # Supported input pairs (both orders).  Value = (CP.xxx_INPUTS, reorder)
    # reorder=True means the user order is the reverse of the pair CoolProp accepts.
    _INPUT_SPEC = {
        "PT": (CP.PT_INPUTS, False),
        "TP": (CP.PT_INPUTS, True),
        "PUmass": (CP.PUmass_INPUTS, False),
        "UmassP": (CP.PUmass_INPUTS, True),
        "DmassP": (CP.DmassP_INPUTS, False),
        "PDmass": (CP.DmassP_INPUTS, True),
        "HmassP": (CP.HmassP_INPUTS, False),
        "PHmass": (CP.HmassP_INPUTS, True),
        "PQ": (CP.PQ_INPUTS, False),
        "QP": (CP.PQ_INPUTS, True),
        "DmassT": (CP.DmassT_INPUTS, False),
        "TDmass": (CP.DmassT_INPUTS, True),
        "DmassUmass": (CP.DmassUmass_INPUTS, False),
        "UmassDmass": (CP.DmassUmass_INPUTS, True),
        "DmassHmass": (CP.DmassHmass_INPUTS, False),
        "HmassDmass": (CP.DmassHmass_INPUTS, True),
        "DmassQ": (CP.DmassQ_INPUTS, False),
        "QDmass": (CP.DmassQ_INPUTS, True),
        "TUmass": (CP.TUmass_INPUTS, False),
        "UmassT": (CP.TUmass_INPUTS, True),
        "HmassT": (CP.HmassT_INPUTS, False),
        "THmass": (CP.HmassT_INPUTS, True),
        "QT": (CP.QT_INPUTS, False),
        "TQ": (CP.QT_INPUTS, True),
        "SmassT": (CP.SmassT_INPUTS, False),
        "TSmass": (CP.SmassT_INPUTS, True),
        "SmassUmass": (CP.SmassUmass_INPUTS, False),
        "UmassSmass": (CP.SmassUmass_INPUTS, True),
        "DmassSmass": (CP.DmassSmass_INPUTS, False),
        "SmassDmass": (CP.DmassSmass_INPUTS, True),
        "HmassSmass": (CP.HmassSmass_INPUTS, False),
        "SmassHmass": (CP.HmassSmass_INPUTS, True),
        "QSmass": (CP.QSmass_INPUTS, False),
        "SmassQ": (CP.QSmass_INPUTS, True),
        "PSmass": (CP.PSmass_INPUTS, False),
        "SmassP": (CP.PSmass_INPUTS, True),
    }

    def __init__(self, library, fluid_name):
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
        # extract properties necessary for initializing abstractstate
        if library == 'CoolProp':
            library = 'HEOS'

        self.FluidName = fluid_name
        self.Library = library
        self._abstract_state = None

        # legacy code. I do not imagine myself putting a fluid name with [1] at the end, but it is in there, so i assume it can be called... 
        name = fluid_name
        if len(name) > 3 and name[-3:] == "[1]":
            name = name[:-3]

        # initalize abstractstate:
        self._abstract_state = AbstractState(self.Library, name)

        # compute critical point properties for the fluid, to be used in the PropsSI method for points that are close to the critical point.
        Tcrit = self._abstract_state.T_critical()
        Dcrit = self._abstract_state.rhomass_critical()
        Pcrit = self._abstract_state.p_critical()
        self.critical_point_vals = (Tcrit, Dcrit, Pcrit)

    @staticmethod
    def _update_wrapper(AS: AbstractState, input_spec: CP.PQ_INPUTS, x: float, y: float, verbose: bool = False) -> bool:
        """
        Coolprop utility to allow nan return upon vectorized evaluation of AbstractState. Note, input_spec
        must not necessarily be CP.PQ_INPUTS, can be other pairs, I wanted to give an example for type hinting.
        """
        try:
            AS.update(input_spec, x, y)
            return AS, False
        except Exception as e:
            if verbose:
                print("Failed to update abstractstate. CoolProp output:", e)
            return AS, True

    def _get_abstract_state(self) -> AbstractState:
        """
        If AbstractState instance is already created for the fluid type and library, no need to create it over and over again.
        """
        if self._abstract_state is None:
            name = self.FluidName
            if len(name) > 3 and name[-3:] == "[1]":
                name = name[:-3]
            self._abstract_state = AbstractState(self.Library, name)
        return self._abstract_state

    def _abstractstate_mass_syntax(self, s: str) -> str:
        """
        Converts PropsSI syntax to AbstractState syntax. for properties that are typically mass-averaged, the subscript mass should be added behind it.
        """
        if s in CoolPropAbstractState_v2._MASS_PROPS:
            return s + "mass"
        return s
               
    def _get_input_spec(self, x_str: str, y_str: str) -> tuple[CP.PQ_INPUTS, bool]:
        """
        Method to convert specified PropsSI input spec into a coolprop inputs object, required for updating the abstractstate thermodynamic state in update_and_get using the coolprop abstractstate
        update method. Note, not really CP.PQ_INPUTS, can be other pairs, but i had to give an example. 
        For all inputs, refer to "input_pairs" section of https://coolprop.org/_static/doxygen/html/namespace_cool_prop.html#aa1ce7c368d1058004293708038241850a648039a97f7392876038eaf56cf91e95

        Attributes
        ----------
        x_str: str
            String corresponding to the first input variable, e.g. "T" for temperature or "P" for pressure.
        y_str: str
            String corresponding to the second input variable, e.g. "T" for temperature or "P" for pressure.
        """
        key = x_str + y_str
        try:
            return self._INPUT_SPEC[key]
        except KeyError:
            raise ValueError(
                f"Unsupported input pair '{x_str}'+'{y_str}'. "
                f"Supported combinations: {sorted(self._INPUT_SPEC)}"
            )

    @staticmethod
    @np.vectorize(otypes=[float])
    def _extract_single(AS: AbstractState, prop_AS: str, input_spec: int, x: float, y: float,
                    reorder: bool, x_str_AS: str, y_str_AS: str, verbose: bool = False):
        """
        Vectorized method to update the AbstractState with the specified input specification and input variables, and return the specified output variable. 
        The method returns nan for points that are not valid for the AbstractState (e.g. points outside the phase envelope).

        Arguments
        ---------
        AS: AbstractState
            CoolProp AbstractState object to update and extract properties from.
        input_spec: int
            CoolProp input specification corresponding to the x_str and y_str variables, e.g. CP.PT_INPUTS for temperature and pressure inputs. This is determined in the get_input_spec method.
        x_str: str
            String corresponding to the first input variable, e.g. "T" for temperature or "P" for pressure.
        x: float
            Value of the first input variable, e.g. temperature or pressure.
        y_str: str
            String corresponding to the second input variable, e.g. "T" for temperature or "
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
        """Extract a property after a successful update."""
        failed = CoolPropAbstractState_v2._update_one(AS, input_spec, x, y, reorder, verbose)
        if failed:
            val = CoolPropAbstractState_v2._try_critical_recovery(
                AS, x_str_AS, x, y_str_AS, y, prop_AS
            )
        else:
            if prop_AS == "drhomassdPcT":
                val = AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iT)
            if prop_AS == "dPdDmass_constUmass":
                val = AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iUmass)
            if prop_AS == "dPdUmass_constDmass":
                val = AS.first_partial_deriv(CP.iP, CP.iUmass, CP.iDmass)

            val = getattr(AS, prop_AS)()  

        if prop_AS == "Q":
            val = min(1.0, max(0.0, val))
        return float(val)

    def _extract_tuple(self, AS: AbstractState, input_spec: int, x: float, y: float,
                            reorder: bool, prop_AS_tuple: tuple,
                            x_str_AS: str, y_str_AS: str, verbose: bool = False) -> tuple:
        failed = self._update_one(AS, input_spec, x, y, reorder, verbose)

        vals = []
        for prop_AS in prop_AS_tuple:
            if failed:
                val = self._try_critical_recovery(AS, x_str_AS, x, y_str_AS, y, prop_AS)
            elif prop_AS == "drhomassdPcT":
                val = AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iT)
            elif prop_AS == "dPdDmass_constUmass":
                val = AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iUmass)
            elif prop_AS == "dPdUmass_constDmass":
                val = AS.first_partial_deriv(CP.iP, CP.iUmass, CP.iDmass)
            else:
                val = getattr(AS, prop_AS)()

            if prop_AS == "Q":
                val = min(1.0, max(0.0, val))
            vals.append(float(val))

        return tuple(vals)

    @staticmethod
    def _update_one(AS: AbstractState, input_spec, x: float, y: float,
                    reorder: bool, verbose: bool = False) -> bool:
        """
        Attempt a single update. Returns True if the update failed
        (caller should return nan or try critical recovery).
        Uses a tiny last-state cache to skip identical consecutive points.
        """
        if reorder:
            xx, yy = y, x
        else:
            xx, yy = x, y

        try:
            AS.update(input_spec, xx, yy)
            return False
        except Exception as e:
            if verbose:
                print("Failed to update AbstractState:", e)
            return True


    def _critical_value(self, AS: AbstractState, prop_str_AS: str, prop_val: float) -> bool:
        """
        This method checks if the specified input value is close to the critical point, 
        and returns True if it is, and False otherwise.

        It does this by calculating the input values specified from the critical point
        T and S values (which were found, from limited testing) to always return a value. 

        Attributes
        ----------
        AS: AbstractState
            AbstractState object to extract critical point properties from.
        prop_str_AS: str
            String corresponding to the property variable, e.g. "T" for temperature or "P" for pressure.
            should comply with CoolProp AbstractState syntax, see self._PropsSI_syntax_to_AbstractState_syntax for translation from PropsSI syntax to AbstractState syntax.
        prop_val: float
            Value of the property variable, e.g. temperature or pressure.

        Returns
        -------
        bool
            True if the specified input pair is close to the critical point, False otherwise.
        """
        Tcrit, Dcrit, _ = self.critical_point_vals
        try:
            AS.update(CP.DmassT_INPUTS, Dcrit, Tcrit)
            S = AS.smass()
            AS.update(CP.SmassT_INPUTS, S, Tcrit)

            if prop_str_AS == "Q":
                prop_crit = AS.Q()
                prop_crit = max(0.0, min(1.0, prop_crit))
            else:
                # Map the AS-style name to the key used by _extract / _getters
                translator = {
                    "Umass": "umass",
                    "Dmass": "rhomass",
                    "Hmass": "hmass",
                    "A": "speed_sound",
                    "T": "T",
                    "P": "p",
                    "Smass": "smass",
                    "Cpmass": "cpmass",
                    "Cvmass": "cvmass",
                    "d(P)/d(D)|T": "drhomassdPcT",
                    "d(P)/d(D)|U": "dPdDmass_constUmass",
                    "d(P)/d(U)|D": "dPdUmass_constDmass",
                    "Phase": "phase",
                    "V": "viscosity",
                }
                key = translator.get(prop_str_AS, prop_str_AS)
                prop_crit = self._extract(AS, key)

            return bool(np.isclose(prop_val, prop_crit, rtol=1e-5, atol=1e-5))
        except Exception:
            return False

    def _try_critical_recovery(self, AS: AbstractState, x_str_AS: str, x: float,
                                   y_str_AS: str, y: float, out_key: str) -> float:
        """
        Called only after a normal update has failed.
        If both inputs are judged to be near the critical point, force the
        state to the critical point and return the requested property.
        Otherwise return nan.
        """
        if not (self._critical_value(AS, x_str_AS, x) and
                self._critical_value(AS, y_str_AS, y)):
            return np.nan

        # Force state to critical point
        Tcrit, Dcrit, _ = self.critical_point_vals
        try:
            AS.update(CP.DmassT_INPUTS, Dcrit, Tcrit)
            S = AS.smass()
            AS.update(CP.SmassT_INPUTS, S, Tcrit)
            return self._extract(AS, out_key)
        except Exception:
            return np.nan

    def PropsSI(self, prop: str | tuple[str], x_str: str = None, x: float | np.ndarray = None, y_str: str = None, y: float | np.ndarray = None, verbose: bool = False) -> float | np.ndarray:
        """
        Integral functionality, uses various methods to convert user input to an input spec accepted by AbstractState syntax, and extracts fluid thermodynamic property according to user specification. 
        using a CoolProp AbstractState syntax. 

        Attributes
        ----------
        prop: str | tuple[str]
            String corresponding to the desired output variable, e.g. "T" for temperature or "P" for pressure.
            May also be a tuple of such strings, e.g. ("T", "P", "Dmass"), to request several properties at the
            same (x, y) point. When a tuple is given, the AbstractState is updated only ONCE per point and every
            requested property is read off that single update, instead of updating once per property.
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
        output: float | np.ndarray | tuple[float | np.ndarray, ...]
            Value of the desired output variable, e.g. temperature or pressure. Will be a float for single point evaluation, or a numpy array for vectorized evaluation. 
            For points that are not valid for the AbstractState (e.g. points outside the phase envelope), nan will be returned.
            If `prop` was a tuple, a tuple of such values is returned instead, one per requested property, in the
            same order as `prop`.
        """
        AS = self._get_abstract_state()

        if isinstance(prop, tuple):
            prop_AS_tuple = tuple(self._abstractstate_mass_syntax(p) for p in prop)
            prop_AS_tuple = tuple(self._TRANSLATOR.get(p, p) for p in prop_AS_tuple)
            x_str_AS = self._abstractstate_mass_syntax(x_str)
            y_str_AS = self._abstractstate_mass_syntax(y_str)
            input_spec, reorder = self._get_input_spec(x_str_AS, y_str_AS)

            extractor = np.vectorize(
                self._extract_tuple,
                otypes=[float] * len(prop_AS_tuple),
                excluded={0, 1, 4, 5, 6, 7, 8},  # everything except x (2) and y (3)
            )
            return extractor(AS, input_spec, x, y, reorder, prop_AS_tuple, x_str_AS, y_str_AS, verbose)

        if isinstance(prop, str):
            if prop in ("Tcrit", "Pcrit", "Dcrit", "Tmax", "M", "Ttriple"):
                AS = self._get_abstract_state()
                if prop == "Tcrit":
                    return self.critical_point_vals[0]
                if prop == "Dcrit":
                    return self.critical_point_vals[1]
                if prop == "Pcrit":
                    return self.critical_point_vals[2]
                if prop == "Tmax":
                    return AS.Tmax()
                if prop == "M":
                    return AS.molar_mass()
                if prop == "Ttriple":
                    return AS.Ttriple()

            prop_AS = self._abstractstate_mass_syntax(prop)
            prop_AS = self._TRANSLATOR.get(prop_AS, prop_AS)

        x_str_AS = self._abstractstate_mass_syntax(x_str)
        y_str_AS = self._abstractstate_mass_syntax(y_str)

        input_spec, reorder = self._get_input_spec(x_str_AS, y_str_AS)

        return self._extract_single(AS, prop_AS, input_spec, x, y, reorder, x_str_AS, y_str_AS, verbose)