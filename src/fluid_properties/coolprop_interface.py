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
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP


class CoolPropAbstractState_v2:
    """
    CoolProp AbstractState wrapper that keeps the familiar PropsSI syntax
    while using AbstractState under the hood. Supports scalar and vectorized
    evaluation. Invalid states return nan.

    Performance notes (relative to the original vectorize + per-point exception version):
    - Explicit loop instead of np.vectorize
    - Class-level lookup tables (no repeated string concat / getattr)
    - Last-state cache (identical consecutive points skip update)
    - Bound method getters for the most common properties
    """

    # ------------------------------------------------------------------
    # Class-level constants (created once)
    # ------------------------------------------------------------------
    _MASS_PROPS = frozenset({"D", "U", "H", "S"})

    # PropsSI-style name  ->  AbstractState method name
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
        "d(P)/d(D)|T": "drhomassdPcT",   # special-cased below
        "Phase": "phase",
        "V": "viscosity",
    }

    # Supported input pairs (both orders).  Value = (CP.xxx_INPUTS, reorder)
    # reorder=True means the user order is the reverse of the CoolProp pair.
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

    def __init__(self, library: str, fluid_name: str):
        if library == "CoolProp":
            library = "HEOS"

        self.FluidName = fluid_name
        self.Library = library
        self._abstract_state = None

        # Strip trailing "[1]" if present (legacy)
        name = fluid_name
        if len(name) > 3 and name[-3:] == "[1]":
            name = name[:-3]

        self._abstract_state = AbstractState(self.Library, name)

        # Critical-point cache
        Tcrit = self._abstract_state.T_critical()
        Dcrit = self._abstract_state.rhomass_critical()
        Pcrit = self._abstract_state.p_critical()
        self.critical_point_vals = (Tcrit, Dcrit, Pcrit)

        # Last successful state cache (avoids redundant updates)
        self._last_spec = None          # CP.xxx_INPUTS
        self._last_x = None
        self._last_y = None
        self._last_ok = False

        # Pre-bind the most common getters so we avoid attribute lookup
        # inside the hot loop.  Special keys are handled separately.
        AS = self._abstract_state
        self._getters = {
            "umass": AS.umass,
            "rhomass": AS.rhomass,
            "hmass": AS.hmass,
            "smass": AS.smass,
            "T": AS.T,
            "p": AS.p,
            "Q": AS.Q,
            "cpmass": AS.cpmass,
            "cvmass": AS.cvmass,
            "gibbsmass": AS.gibbsmass,
            "speed_sound": AS.speed_sound,
            "viscosity": AS.viscosity,
            "phase": AS.phase,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_abstract_state(self) -> AbstractState:
        if self._abstract_state is None:
            name = self.FluidName
            if len(name) > 3 and name[-3:] == "[1]":
                name = name[:-3]
            self._abstract_state = AbstractState(self.Library, name)
            # re-bind getters after recreation
            AS = self._abstract_state
            self._getters = {
                "umass": AS.umass,
                "rhomass": AS.rhomass,
                "hmass": AS.hmass,
                "smass": AS.smass,
                "T": AS.T,
                "p": AS.p,
                "Q": AS.Q,
                "cpmass": AS.cpmass,
                "cvmass": AS.cvmass,
                "gibbsmass": AS.gibbsmass,
                "speed_sound": AS.speed_sound,
                "viscosity": AS.viscosity,
                "phase": AS.phase,
            }
        return self._abstract_state

    @staticmethod
    def _to_AS_name(s: str) -> str:
        """PropsSI short name -> AbstractState-style name used in tables."""
        if s in CoolPropAbstractState_v2._MASS_PROPS:
            return s + "mass"
        return s

    def _get_input_spec(self, x_str: str, y_str: str):
        key = x_str + y_str
        try:
            return self._INPUT_SPEC[key]
        except KeyError:
            raise ValueError(
                f"Unsupported input pair '{x_str}'+'{y_str}'. "
                f"Supported combinations: {sorted(self._INPUT_SPEC)}"
            )

    def _update_one(self, AS: AbstractState, input_spec, x: float, y: float,
                    reorder: bool, verbose: bool = False) -> bool:
        """
        Attempt a single update. Returns True if the update failed
        (caller should return nan).
        Uses a tiny last-state cache to skip identical consecutive points.
        """
        if reorder:
            xx, yy = y, x
        else:
            xx, yy = x, y

        # Cache hit?
        if (self._last_ok and
                self._last_spec is input_spec and
                self._last_x == xx and
                self._last_y == yy):
            return False

        try:
            AS.update(input_spec, xx, yy)
            self._last_spec = input_spec
            self._last_x = xx
            self._last_y = yy
            self._last_ok = True
            return False
        except Exception as e:
            self._last_ok = False
            if verbose:
                print("Failed to update AbstractState:", e)
            return True

    def _extract(self, AS: AbstractState, prop_AS: str):
        """Extract a property after a successful update."""
        if prop_AS == "drhomassdPcT":
            return AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iT)
        getter = self._getters.get(prop_AS)
        if getter is not None:
            return getter()
        # fallback (should be rare)
        return getattr(AS, prop_AS)()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def PropsSI(self, prop: str,
                x_str: str = None, x=None,
                y_str: str = None, y=None,
                verbose: bool = False):
        """
        PropsSI-compatible interface backed by AbstractState.

        Parameters
        ----------
        prop : str
            Desired output (e.g. "D", "H", "T", "P", "Q", ...).
        x_str, y_str : str
            Input variable names (PropsSI style).
        x, y : float or array-like
            Input values.  Scalars or 1-D arrays of the same length.
        verbose : bool
            Print CoolProp exception messages on failure.

        Returns
        -------
        float or np.ndarray
            Requested property.  Invalid states become nan.
        """
        # ---- trivial / state-independent properties --------------------
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

        # ---- normal flash ----------------------------------------------
        AS = self._get_abstract_state()

        prop_AS = self._to_AS_name(prop)
        x_str_AS = self._to_AS_name(x_str)
        y_str_AS = self._to_AS_name(y_str)

        input_spec, reorder = self._get_input_spec(x_str_AS, y_str_AS)

        # Map to the internal method name used by _extract
        out_key = self._TRANSLATOR.get(prop_AS, prop_AS)

        # Normalise inputs to 1-D arrays (keeps scalar fast-path simple)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        n = x_arr.size
        if y_arr.size != n:
            raise ValueError("x and y must have the same number of elements")

        out = np.empty(n, dtype=float)

        for i in range(n):
            failed = self._update_one(AS, input_spec, x_arr[i], y_arr[i],
                                      reorder, verbose)
            if failed:
                out[i] = np.nan
            else:
                out[i] = self._extract(AS, out_key)

        # Quality clipping (match original behaviour)
        if prop_AS == "Q":
            np.clip(out, 0.0, 1.0, out=out)

        # Return scalar when both inputs were scalar
        if np.isscalar(x) and np.isscalar(y):
            return float(out[0])
        return out



