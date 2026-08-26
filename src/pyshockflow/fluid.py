import sys

import numpy as np
import fluid_properties.fluid_properties as FP
from functools import partial, lru_cache
from scipy.optimize import fsolve

class FluidIdeal():
    """
    Ideal Fluid Class, where thermodynamic properties and transformation are computed with ideal gas laws
    """
    def __init__(self, gmma, Rgas, mu = None):
        self.gmma = gmma
        self.Rgas = Rgas
        self.mu = mu

    def computeTemperature_p_rho(self, p, rho):
        return (p/rho)/self.Rgas

    def computeDensity_p_T(self, p, T):
        return p/self.Rgas/T

    def computePressure_rho_e(self, rho, e):
        return (self.gmma-1)*rho*e

    def computeInternalEnergy_p_rho(self, p, rho):
        return (p / (self.gmma - 1) / rho)

    def computeTotalInternalEnergy_Tt(self, Tt):
        return self.Rgas*Tt/(self.gmma-1)

    def computeEntropy_p_rho(self, p, rho):
        return p/(rho**self.gmma)



    def computeSoundSpeed_p_rho(self, p, rho):
        return np.sqrt(self.gmma*p/rho)

    def computeMach_u_p_rho(self, u, p, rho):
        soundSpeed = self.computeSoundSpeed_p_rho(p, rho)
        return np.abs(u)/soundSpeed

    def computeComprFactorZ_p_rho(self, p, rho):
        """
        Includes rho since real fluids need this information. We want to achieve a simple 
        logic in driver.py. We can achieve this by naming the property extraction functions
        for ideal and real fluids the same, such that we can call a function, and it exists
        for both. However, the ideal fluid does not need the density to compute the 
        compressibility factor, but we still include it in the function signature to keep 
        the same function signature as the real fluid. This is why the density is still 
        present as an input.
        """
        if isinstance(p, np.ndarray):
            return 1+np.zeros_like(p)
        else:
            return 1

    def computeFunDerGamma_p_rho(self, p, rho):
        """Density present for same reason as computeComprFactorZ_p_rho."""
        if isinstance(p, np.ndarray):
            return 0.5*(self.gmma+1)+np.zeros_like(p)
        else:
            return 0.5*(self.gmma+1)

    def compute_gammapv_p_rho(self, p, rho):
        """Density present for same reason as computeComprFactorZ_p_rho."""
        if isinstance(p, np.ndarray):
            gmma_pv = np.zeros_like(p)+self.gmma
        else:
            gmma_pv = self.gmma
        return gmma_pv



    def computeDensityIsentropic_p1_p2_rho1(self, p1, p2, rho1):
        return rho1*(p2/p1)**(1/self.gmma)

    def computeTotalPressure_p_M(self, p, M):
        return p*(1+(self.gmma-1)/2*M**2)**(self.gmma/(self.gmma-1))

    def computeTotalTemperature_T_M(self, T, M):
        return T*(1+(self.gmma-1)/2*M**2)

    def computeTemperature_Tt_M(self, Tt, M):
        return Tt/(1+(self.gmma-1)/2*M**2)

    def computePressure_Pt_M(self, Pt, M):
        return Pt/((1+(self.gmma-1)/2*M**2)**(self.gmma/(self.gmma-1)))

    def computeMach_pt_p(self, pt, p):
        mach = np.sqrt( 2/(self.gmma-1) * ((pt/p)**((self.gmma-1)/self.gmma)-1) )
        return mach



    def computeInletQuantitiesTotal_pt_Tt(self, pressure, totPressure, totTemperature, direction):
        mach = self.computeMach_pt_p(totPressure, pressure)
        temperature = self.computeTemperature_Tt_M(totTemperature, mach)
        density = self.computeDensity_p_T(pressure, temperature)
        soundSpeed = self.computeSoundSpeed_p_rho(pressure, density)
        velocity = mach*soundSpeed*direction
        energy = self.computeInternalEnergy_p_rho(pressure, density)
        return density, velocity, energy

    def computeInletQuantitiesTotal_pt_Q(self, pressure, totPressure, quality, direction):
        """non-used inputs present for same reason as computeComprFactorZ_p_rho."""
        raise NotImplementedError("Two-phase flow is not supported for ideal fluids.")

    def computeInletQuantitiesStatic_p_T(self, pressure, temperature):
        density = self.computeDensity_p_T(pressure, temperature)
        energy = self.computeInternalEnergy_p_rho(pressure, density)
        return density, energy

    def computeInletQuantitiesStatic_p_Q(self, pressure, quality):
        """non-used inputs present for same reason as computeComprFactorZ_p_rho."""
        raise NotImplementedError("Two-phase flow is not supported for ideal fluids.")



    def computeChiKappa_VinokurScheme_p_rho(self, p, rho):
        chi = 0
        kappa = self.gmma-1
        return chi, kappa


import numpy as np
from functools import lru_cache, partial
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
import sys

class FluidReal():
    """
    Real Fluid Class, where thermodynamic properties and transformations are taken from CoolProp
    """
    def __init__(self, fluid_name, fluid_library, fluid_property_extraction_method, print_error=True):
        self.fluid_name = fluid_name
        self.fluid_library = fluid_library if fluid_library != 'CoolProp' else 'HEOS'
        self.extraction_method = fluid_property_extraction_method.lower()
        
        # Store the fluid identifier for PropsSI calls
        self._fluid_string = fluid_name  # Always keep the string name for PropsSI
        
        if self.extraction_method == 'fluid':
            # Use string-based PropsSI (original CoolProp high-level interface)
            self.fluid = self._fluid_string
        elif self.extraction_method == 'abstractstate':
            # Use AbstractState directly
            self._abstract_state = AbstractState(self.fluid_library, fluid_name)
            self.fluid = self._abstract_state
        elif self.extraction_method == 'abstractstate_v2':
            # Use optimized AbstractState wrapper
            from fluid_properties.coolprop_interface import CoolPropAbstractState_v2  # Adjust import
            self.fluid = CoolPropAbstractState_v2(self.fluid_library, fluid_name)
        elif self.extraction_method == 'abstractstate_v3':
            # Use the newly optimized version
            from fluid_properties.coolprop_interface import CoolPropAbstractState_v3  # Adjust import
            self.fluid = CoolPropAbstractState_v3(self.fluid_library, fluid_name)
        else:
            raise ValueError(f"Unknown fluid property extraction method: {fluid_property_extraction_method}")

        # Cache scalar property calls
        self._computeInternalEnergyScalarCached = lru_cache(maxsize=200000)(
            self._computeInternalEnergyScalar
        )
        self._computePressureScalarCached = lru_cache(maxsize=200000)(
            self._computePressureScalar
        )
        self._computeSoundSpeedScalarCached = lru_cache(maxsize=200000)(
            self._computeSoundSpeedScalar
        )

    def _get_property(self, prop, x_str, x, y_str, y):
        """
        Unified property extraction that works with all fluid types
        """
        if self.extraction_method == 'fluid':
            # Use string-based PropsSI
            return CP.PropsSI(prop, x_str, x, y_str, y, self._fluid_string)
        elif self.extraction_method == 'abstractstate':
            # Use AbstractState directly
            return self._get_property_abstractstate(prop, x_str, x, y_str, y)
        elif self.extraction_method in ['abstractstate_v2', 'abstractstate_v3']:
            # Use the wrapper's PropsSI method
            return self.fluid.PropsSI(prop, x_str, x, y_str, y)
    
    def _get_property_abstractstate(self, prop, x_str, x, y_str, y):
        """
        Property extraction using AbstractState directly
        """
        # Map PropsSI names to AbstractState methods
        prop_map = {
            'T': 'T', 'P': 'p', 'D': 'rhomass', 'U': 'umass',
            'H': 'hmass', 'S': 'smass', 'Q': 'Q', 'A': 'speed_sound',
            'Cpmass': 'cpmass', 'Cvmass': 'cvmass', 'V': 'viscosity',
            'Phase': 'phase', 'Z': 'compressibility_factor',
            'd(P)/d(D)|T': 'first_partial_deriv'
        }
        
        # Map input pairs to AbstractState input specifications
        input_map = {
            ('P', 'T'): CP.PT_INPUTS, ('T', 'P'): CP.PT_INPUTS,
            ('P', 'D'): CP.DmassP_INPUTS, ('D', 'P'): CP.DmassP_INPUTS,
            ('P', 'H'): CP.HmassP_INPUTS, ('H', 'P'): CP.HmassP_INPUTS,
            ('P', 'Q'): CP.PQ_INPUTS, ('Q', 'P'): CP.PQ_INPUTS,
            ('P', 'S'): CP.PSmass_INPUTS, ('S', 'P'): CP.PSmass_INPUTS,
            ('P', 'U'): CP.PUmass_INPUTS, ('U', 'P'): CP.PUmass_INPUTS,
            ('D', 'T'): CP.DmassT_INPUTS, ('T', 'D'): CP.DmassT_INPUTS,
            ('D', 'U'): CP.DmassUmass_INPUTS, ('U', 'D'): CP.DmassUmass_INPUTS,
            ('D', 'H'): CP.DmassHmass_INPUTS, ('H', 'D'): CP.DmassHmass_INPUTS,
            ('T', 'Q'): CP.QT_INPUTS, ('Q', 'T'): CP.QT_INPUTS,
            ('T', 'S'): CP.SmassT_INPUTS, ('S', 'T'): CP.SmassT_INPUTS,
        }
        
        # Determine if inputs need reordering
        input_key = (x_str, y_str)
        if input_key in input_map:
            input_spec = input_map[input_key]
            reorder = False
        else:
            input_key_reversed = (y_str, x_str)
            if input_key_reversed in input_map:
                input_spec = input_map[input_key_reversed]
                reorder = True
            else:
                raise ValueError(f"Unsupported input pair: {x_str}, {y_str}")
        
        # Update state
        try:
            if reorder:
                self._abstract_state.update(input_spec, y, x)
            else:
                self._abstract_state.update(input_spec, x, y)
        except:
            return np.nan
        
        # Get property
        if prop == 'd(P)/d(D)|T':
            return self._abstract_state.first_partial_deriv(CP.iP, CP.iDmass, CP.iT)
        elif prop in prop_map:
            method = getattr(self._abstract_state, prop_map[prop])
            return method()
        else:
            raise ValueError(f"Unsupported property: {prop}")

    def clearPropertyCaches(self):
        """Clear all scalar-property caches."""
        self._computeInternalEnergyScalarCached.cache_clear()
        self._computePressureScalarCached.cache_clear()
        self._computeSoundSpeedScalarCached.cache_clear()

    def _computeInternalEnergyScalar(self, p, rho):
        return self._get_property('U', 'P', p, 'D', rho)

    def _computePressureScalar(self, rho, e):
        return self._get_property('P', 'D', rho, 'U', e)

    def _computeSoundSpeedScalar(self, p, rho):
        # This is the critical fix - avoid recursive calls
        if self.extraction_method == 'fluid':
            return self._computeSoundSpeed_p_rho_single_fluid(p, rho)
        else:
            return self._computeSoundSpeed_p_rho_single(p, rho)

    @staticmethod
    def _computeSoundSpeed_p_rho_single_fluid(p, rho, fluid):
        """Sound speed calculation for string-based fluid"""
        phase = CP.PropsSI("Phase", "P", p, "D", rho, fluid)
        
        if phase == 6:  # Two-phase
            # Use HEM model
            T = CP.PropsSI("T", "P", p, "D", rho, fluid)
            y_V = CP.PropsSI("Q", "P", p, "D", rho, fluid)
            y_L = 1 - y_V
            soundSpeed_L = CP.PropsSI("A", "P", p, "Q", 0, fluid)
            soundSpeed_V = CP.PropsSI("A", "P", p, "Q", 1, fluid)
            rho_L = CP.PropsSI("D", "P", p, "Q", 0, fluid)
            rho_V = CP.PropsSI("D", "P", p, "Q", 1, fluid)
            c_p_L = CP.PropsSI("Cpmass", "P", p, "Q", 0, fluid)
            c_p_V = CP.PropsSI("Cpmass", "P", p, "Q", 1, fluid)
            alpha_V = y_V * (rho/rho_V)
            alpha_L = y_L * (rho/rho_L)
            
            ds_dp_cQ_L = (CP.PropsSI("S", "P", p + 1e3, "Q", 0, fluid) -
                          CP.PropsSI("S", "P", p - 1e3, "Q", 0, fluid)) / (2 * 1e3)
            ds_dp_cQ_V = (CP.PropsSI("S", "P", p + 1e3, "Q", 1, fluid) -
                          CP.PropsSI("S", "P", p - 1e3, "Q", 1, fluid)) / (2 * 1e3)
            
            a = (rho * (
                    alpha_L / (rho_L * soundSpeed_L**2) +
                    alpha_V / (rho_V * soundSpeed_V**2) +
                    T * ((alpha_L * rho_L / c_p_L) * ds_dp_cQ_L**2 +
                         (alpha_V * rho_V / c_p_V) * ds_dp_cQ_V**2)
                    ))**(-0.5)
            return a
        else:
            # Single phase
            return CP.PropsSI("A", "P", p, "D", rho, fluid)

    def _computeSoundSpeed_p_rho_single(self, p, rho):
        """Sound speed calculation for AbstractState-based fluids"""
        if self.extraction_method in ['abstractstate_v2', 'abstractstate_v3']:
            # Use the wrapper's PropsSI method
            phase = self.fluid.PropsSI("Phase", "P", p, "D", rho)
        else:
            # Use AbstractState directly
            phase = self._get_property('Phase', 'P', p, 'D', rho)
        
        if phase == 6:  # Two-phase
            # Similar HEM model but using the appropriate property extraction
            T = self._get_property('T', 'P', p, 'D', rho)
            y_V = self._get_property('Q', 'P', p, 'D', rho)
            y_L = 1 - y_V
            
            if self.extraction_method in ['abstractstate_v2', 'abstractstate_v3']:
                soundSpeed_L = self.fluid.PropsSI("A", "P", p, "Q", 0)
                soundSpeed_V = self.fluid.PropsSI("A", "P", p, "Q", 1)
                rho_L = self.fluid.PropsSI("D", "P", p, "Q", 0)
                rho_V = self.fluid.PropsSI("D", "P", p, "Q", 1)
                c_p_L = self.fluid.PropsSI("Cpmass", "P", p, "Q", 0)
                c_p_V = self.fluid.PropsSI("Cpmass", "P", p, "Q", 1)
            else:
                soundSpeed_L = self._get_property('A', 'P', p, 'Q', 0)
                soundSpeed_V = self._get_property('A', 'P', p, 'Q', 1)
                rho_L = self._get_property('D', 'P', p, 'Q', 0)
                rho_V = self._get_property('D', 'P', p, 'Q', 1)
                c_p_L = self._get_property('Cpmass', 'P', p, 'Q', 0)
                c_p_V = self._get_property('Cpmass', 'P', p, 'Q', 1)
            
            alpha_V = y_V * (rho/rho_V)
            alpha_L = y_L * (rho/rho_L)
            
            # For AbstractState, we need to use finite differences differently
            ds_dp_cQ_L = (self._get_property('S', 'P', p + 1e3, 'Q', 0) -
                          self._get_property('S', 'P', p - 1e3, 'Q', 0)) / (2 * 1e3)
            ds_dp_cQ_V = (self._get_property('S', 'P', p + 1e3, 'Q', 1) -
                          self._get_property('S', 'P', p - 1e3, 'Q', 1)) / (2 * 1e3)
            
            a = (rho * (
                    alpha_L / (rho_L * soundSpeed_L**2) +
                    alpha_V / (rho_V * soundSpeed_V**2) +
                    T * ((alpha_L * rho_L / c_p_L) * ds_dp_cQ_L**2 +
                         (alpha_V * rho_V / c_p_V) * ds_dp_cQ_V**2)
                    ))**(-0.5)
            return a
        else:
            # Single phase
            return self._get_property('A', 'P', p, 'D', rho)

    # Rest of the methods remain the same but use _get_property instead of FP.PropsSI
    def computeTemperature_p_rho(self, p, rho):
        return self._get_property('T', 'P', p, 'D', rho)

    def computeTemperature_p_Q(self, p, Q):
        return self._get_property('T', 'P', p, 'Q', Q)

    def computeTemperature_p_S(self, p, s):
        return self._get_property('T', 'P', p, 'S', s)

    def computeDensity_p_T(self, p, T):
        return self._get_property('D', 'P', p, 'T', T)

    def computeDensity_p_S(self, p, s):
        return self._get_property('D', 'P', p, 'S', s)

    def computeDensity_p_s(self, p, s):
        return self._get_property('D', 'P', p, 'S', s)

    def computeDensity_p_h(self, p, h):
        return self._get_property('D', 'P', p, 'H', h)

    def computePressure_rho_e(self, rho, e):
        rho_arr = np.asarray(rho, dtype=float)
        e_arr = np.asarray(e, dtype=float)
        rho_arr, e_arr = np.broadcast_arrays(rho_arr, e_arr)
        return np.vectorize(self._computePressureScalarCached, otypes=[float])(rho_arr, e_arr)

    def computeInternalEnergy_p_rho(self, p, rho):
        p_arr = np.asarray(p, dtype=float)
        rho_arr = np.asarray(rho, dtype=float)
        p_arr, rho_arr = np.broadcast_arrays(p_arr, rho_arr)
        return np.vectorize(self._computeInternalEnergyScalarCached, otypes=[float])(p_arr, rho_arr)

    def computeInternalEnergy_p_T(self, p, T):
        return self._get_property('U', 'P', p, 'T', T)

    def computeInternalEnergy_p_Q(self, p, Q):
        return self._get_property('U', 'P', p, 'Q', Q)

    def computeInternalEnergy_p_s(self, p, s):
        return self._get_property('U', 'P', p, 'S', s)

    def computeEntropy_p_rho(self, p, rho):
        return self._get_property('S', 'P', p, 'D', rho)

    def computeEntropy_p_T(self, p, T):
        return self._get_property('S', 'P', p, 'T', T)

    def computeEntropy_p_Q(self, p, Q):
        return self._get_property('S', 'P', p, 'Q', Q)

    def computeEnthalpy_p_rho(self, p, rho):
        return self._get_property('H', 'P', p, 'D', rho)

    def computeEnthalpy_p_T(self, p, T):
        return self._get_property('H', 'P', p, 'T', T)

    def computeEnthalpy_p_Q(self, p, Q):
        return self._get_property('H', 'P', p, 'Q', Q)

    def computeEnthalpy_p_s(self, p, s):
        return self._get_property('H', 'P', p, 'S', s)

    def computeQuality_p_rho(self, p, rho):
        return self._get_property('Q', 'P', p, 'D', rho)

    def computeSoundSpeed_p_rho(self, p, rho):
        p_arr = np.asarray(p, dtype=float)
        rho_arr = np.asarray(rho, dtype=float)
        p_arr, rho_arr = np.broadcast_arrays(p_arr, rho_arr)
        return np.vectorize(self._computeSoundSpeedScalarCached, otypes=[float])(p_arr, rho_arr)

    def computeMach_u_p_rho(self, u, p, rho):
        soundSpeed = self.computeSoundSpeed_p_rho(p, rho)
        return np.abs(u)/soundSpeed

    def computeComprFactorZ_p_rho(self, p, rho):
        return self._get_property('Z', 'P', p, 'D', rho)

    def computeFunDerGamma_p_rho(self, p, rho):
        try:
            return self._get_property("FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS", "P", p, "D", rho)
        except:
            T = self.computeTemperature_p_rho(p, rho)
            try:
                Q = self._get_property("Q", "T", T, "P", p)
            except:
                Q = 1

            G_liquid = self._get_property("FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS", "T", T, "Q", 0)
            G_vapor = self._get_property("FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS", "T", T, "Q", 1)

            G = (1 - Q) * G_liquid + Q * G_vapor
            return G

    def compute_gammapv_p_rho(self, p, rho):
        cp = self._get_property("Cpmass", "P", p, "D", rho)
        cv = self._get_property("Cvmass", "P", p, "D", rho)
        dp_drho_T = self._get_property("d(P)/d(D)|T", "P", p, "D", rho)
        dp_dv_T = - rho**2 * dp_drho_T
        gmma_pv = -1/(p*rho) * cp/cv * dp_dv_T
        return gmma_pv

    def compute_gammapt_p_T(self, p, T):
        rho = self._get_property("D", "P", p, "T", T)
        d_rho_dT_P = self._get_property("d(D)/d(T)|P", "P", p, "T", T)
        dv_dT_P = - d_rho_dT_P / (rho**2)
        cp = self._get_property("Cpmass", "P", p, "T", T)
        gamma_pT = 1 / (1 - p/cp*dv_dT_P)
        return gamma_pT

    def computeDynamicViscosity_p_rho(self, p, rho):
        p = np.asarray(p, dtype=float)
        rho = np.asarray(rho, dtype=float)
        p, rho = np.broadcast_arrays(p, rho)
        vectorized_func = np.vectorize(
            partial(self._computeDynamicViscosity_p_rho_single),
            otypes=[float]
        )
        return vectorized_func(p, rho)

    def _computeDynamicViscosity_p_rho_single(self, p, rho):
        phase = self._get_property("Phase", "P", p, "D", rho)
        
        if phase == 6:  # Two-phase
            y_V = self._get_property("Q", "P", p, "D", rho)
            rho_V = self._get_property("D", "P", p, "Q", 1)
            alpha_V = y_V * rho / rho_V
            mu_V = self._get_property("V", "P", p, "Q", 1)
            mu_L = self._get_property("V", "P", p, "Q", 0)
            mu_2phase = alpha_V * mu_V + (1-alpha_V) * (1+2.5*alpha_V) * mu_L
            return mu_2phase
        else:
            return self._get_property("V", "P", p, "D", rho)

    def computeMach_pt_p_gammapv(self, pt, p, gamma_pv):
        mach = np.sqrt(2/(gamma_pv-1) * ((pt/p)**((gamma_pv-1)/gamma_pv) - 1))
        return mach

    def computeInletQuantitiesTotal_pt_Tt(self, pressure, totPressure, totTemperature, direction):
        entropyTotal = self.computeEntropy_p_T(totPressure, totTemperature)
        density = self.computeDensity_p_S(pressure, entropyTotal)
        enthalpyTotal = self.computeEnthalpy_p_T(totPressure, totTemperature)
        enthalpyStatic = self.computeEnthalpy_p_rho(pressure, density)
        velocity = direction * np.sqrt(2 * (enthalpyTotal - enthalpyStatic))
        energy = self.computeInternalEnergy_p_rho(pressure, density)
        return density, velocity, energy

    def computeInletQuantitiesTotal_pt_Q(self, pressure, totPressure, quality, direction):
        entropyTotal = self.computeEntropy_p_Q(totPressure, quality)
        density = self.computeDensity_p_S(pressure, entropyTotal)
        enthalpyTotal = self.computeEnthalpy_p_Q(totPressure, quality)
        enthalpyStatic = self.computeEnthalpy_p_rho(pressure, density)
        velocity = direction * np.sqrt(2 * (enthalpyTotal - enthalpyStatic))
        energy = self.computeInternalEnergy_p_rho(pressure, density)
        return density, velocity, energy

    def computeInletQuantitiesStatic_p_T(self, pressure, temperature):
        density = self.computeDensity_p_T(pressure, temperature)
        energy = self.computeInternalEnergy_p_T(pressure, temperature)
        return density, energy

    def computeInletQuantitiesStatic_p_Q(self, pressure, quality):
        density = self.computeDensity_p_Q(pressure, quality)
        energy = self.computeInternalEnergy_p_Q(pressure, quality)
        return density, energy

    def computeChiKappa_VinokurScheme_p_rho(self, p, rho):
        e = self._get_property("U", "P", p, "D", rho)
        dp_drho_econst = self._get_property("d(P)/d(D)|U", "P", p, "D", rho)
        dp_de_rhoconst = self._get_property("d(P)/d(U)|D", "P", p, "D", rho)
        chi = dp_drho_econst - e/rho * dp_de_rhoconst
        kappa = dp_de_rhoconst / rho
        return chi, kappa