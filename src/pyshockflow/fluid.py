import sys

import numpy as np
import fluid_properties.fluid_properties as FP
from functools import partial
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


class FluidReal():
    """
    Real Fluid Class, where thermodynamic properties and transformations are taken from coolprop
    """
    def __init__(self, fluid_name, fluid_library, fluid_property_extraction_method, print_error=True):
        self.fluid_name = fluid_name
        self.fluid_library = fluid_library
        if fluid_property_extraction_method.lower() == 'fluid':
            self.fluid = FP.fluid(fluid_library, fluid_name,  print_error=print_error)
        if fluid_property_extraction_method.lower() == 'abstractstate':
            self.fluid = FP.AbstractState(fluid_library, fluid_name)
        if fluid_property_extraction_method.lower() == 'abstractstate_v2':
            self.fluid = FP.AbstractState_v2(fluid_library, fluid_name)



    def computeTemperature_p_rho(self, p, rho):
        T = FP.PropsSI('T', 'P', p, 'D', rho, self.fluid)
        return T

    def computeTemperature_p_Q(self, p, Q):
        T = FP.PropsSI('T', 'P', p, 'Q', Q, self.fluid)
        return T

    def computeTemperature_p_S(self, p, s):
        T = FP.PropsSI('T', 'P', p, 'S', s, self.fluid)
        return T

    def computeDensity_p_T(self, p, T):
        rho = FP.PropsSI('D', 'P', p, 'T', T, self.fluid)
        return rho

    def computeDensity_p_S(self, p, s):
        rho = FP.PropsSI('D', 'P', p, 'S', s, self.fluid)
        return rho

    def computeDensity_p_s(self, p, s):
        return FP.PropsSI('D', 'P', p, 'S', s, self.fluid)

    def computeDensity_p_h(self, p, h):
        return FP.PropsSI('D', 'P', p, 'H', h, self.fluid)

    def computePressure_rho_e(self, rho, e):
        p = FP.PropsSI('P', 'D', rho, 'U', e, self.fluid)
        return p

    def computeInternalEnergy_p_rho(self, p, rho):
        e = FP.PropsSI('U', 'P', p, 'D', rho, self.fluid)
        return e

    def computeInternalEnergy_p_T(self, p, T):
        e = FP.PropsSI('U', 'P', p, 'T', T, self.fluid)
        return e

    def computeInternalEnergy_p_Q(self, p, Q):
        e = FP.PropsSI('U', 'P', p, 'Q', Q, self.fluid)
        return e

    def computeInternalEnergy_p_s(self, p, s):
        e = FP.PropsSI('U', 'P', p, 'S', s, self.fluid)
        return e

    def computeEntropy_p_rho(self, p, rho):
        s = FP.PropsSI('S', 'P', p, 'D', rho, self.fluid)
        return s

    def computeEntropy_p_T(self, p, T):
        s = FP.PropsSI('S', 'P', p, 'T', T, self.fluid)
        return s

    def computeEntropy_p_Q(self, p, Q):
        s = FP.PropsSI('S', 'P', p, 'Q', Q, self.fluid)
        return s

    def computeEnthalpy_p_rho(self, p, rho):
        h = FP.PropsSI('H', 'P', p, 'D', rho, self.fluid)
        return h

    def computeEnthalpy_p_T(self, p, T):
        return FP.PropsSI('H', 'P', p, 'T', T, self.fluid)

    def computeEnthalpy_p_Q(self, p, Q):
        h = FP.PropsSI('H', 'P', p, 'Q', Q, self.fluid)
        return h

    def computeEnthalpy_p_s(self, p, s):
        return FP.PropsSI('H', 'P', p, 'S', s, self.fluid)

    def computeQuality_p_rho(self, p, rho):
        Q = FP.PropsSI('Q', 'P', p, 'D', rho, self.fluid)
        return Q


    def computeSoundSpeed_p_rho(self, p: float | np.ndarray, rho: float | np.ndarray) -> float | np.ndarray:
        # Ensure inputs are numpy arrays
        p = np.asarray(p, dtype=float)
        rho = np.asarray(rho, dtype=float)
        p, rho = np.broadcast_arrays(p, rho)

        # function vectorized separately to allow for both single 
        # values and arrays and for both to still pass the phase 
        # check. Vectorize the core function, passing self.fluid
        vectorized_func = np.vectorize(
            partial(self._computeSoundSpeed_p_rho_single, fluid=self.fluid),
            otypes=[float]
        )
        return vectorized_func(p, rho)

    @staticmethod
    def _computeSoundSpeed_p_rho_single(p: float, rho: float, fluid: str) -> float:
        """single thdy point evaluation"""
        # check if the state is two phase
        # readers can find interpretation of the phase number in the CoolProp documentation:
        # https://coolprop.org/_static/doxygen/html/namespace_cool_prop.html#aa1ce7c368d1058004293708038241850a648039a97f7392876038eaf56cf91e95
        # under section "phases"
        phase = FP.PropsSI("Phase", "P", p, "D", rho, fluid)
        
        # if phase == 6, fluid is in two-phase region. 
        two_phase = False
        if phase == 6:
            two_phase = True

        def _computeSoundSpeed_p_rho_single_phase(p: float, rho: float) -> float:
            a = FP.PropsSI("A", "P", p, "D", rho, fluid)      
            return a
        
        def _computeSoundSpeed_p_rho_two_phase(p: float, rho: float) -> float:
            # two-phase (HEM model from Cioffi et al.)
            T = FP.PropsSI("T", "P", p, "D", rho, fluid)
            y_V = FP.PropsSI("Q", "P", p, "D", rho, fluid)
            y_L = 1 - y_V
            soundSpeed_L = FP.PropsSI("A", "P", p, "Q", 0, fluid)
            soundSpeed_V = FP.PropsSI("A", "P", p, "Q", 1, fluid)
            rho_L = FP.PropsSI("D", "P", p, "Q", 0, fluid)
            rho_V = FP.PropsSI("D", "P", p, "Q", 1, fluid)
            c_p_L = FP.PropsSI("Cpmass", "P", p, "Q", 0, fluid)
            c_p_V = FP.PropsSI("Cpmass", "P", p, "Q", 1, fluid)
            alpha_V = y_V * (rho/rho_V)
            alpha_L = y_L * (rho/rho_L)
            
            # Central difference for ds/dp at constant Q
            ds_dp_cQ_L = (FP.PropsSI("S", "P", p + 1e3, "Q", 0, fluid) -
                            FP.PropsSI("S", "P", p - 1e3, "Q", 0, fluid)) / (2 * 1e3)
            ds_dp_cQ_V = (FP.PropsSI("S", "P", p + 1e3, "Q", 1, fluid) -
                            FP.PropsSI("S", "P", p - 1e3, "Q", 1, fluid)) / (2 * 1e3)

            # Sound speed according to Eq. 29 (Cioffi et al.)
            a = (rho * (
                    alpha_L / (rho_L * soundSpeed_L**2) +
                    alpha_V / (rho_V * soundSpeed_V**2) +
                    T * ((alpha_L * rho_L / c_p_L) * ds_dp_cQ_L**2 +
                            (alpha_V * rho_V / c_p_V) * ds_dp_cQ_V**2)
                    ))**(-0.5)
            return a
        if not two_phase:
            # from tests performed in pyshockflow of this function, when computesoundspeed
            # is called in any region other than two-phase near the two-phase dome, 
            # the value is stable. Values inside the two-phase dome (phase == 6) near the 
            # dome can return -9999980 or nan. 
            a = _computeSoundSpeed_p_rho_single_phase(p, rho)
            # common errors:
            if abs(a) > 99999:
                # for qualities near 0, I did see two-phase SOS in the thousands, but that
                # is considered acceptably finite for an edge case.
                print(f"Warning: Computed sound speed {a} is unusually high for p={p}, rho={rho}.\n"
                    "This issue is common when the thdy pair is considered two-phase by CoolProp\n"
                    "but is nevertheless evaluated using PropsSI, which from experience only\n"
                    "returns sensible values for non-two-phase regions.")
            if np.isnan(a):
                print(f"Warning: Computed sound speed is NaN for p={p}, rho={rho}.\n"
                    "This issue is common when the thdy pair is close to the critical point.\n"
                    "The user may try relaxing the tolerance of the CoolPropAbstractState_v2\n"
                    "_critical_value method. However if the relaxation required is too large\n"
                    "it is recommended to launch a separate investigation.")
            return a
        else:
            # but if the value is computed using the cioffi equation, the returned value
            # has no risk of being -9999980 or nan either, so we can be ensured about stability.
            a = _computeSoundSpeed_p_rho_two_phase(p, rho)
            return a

    def computeMach_u_p_rho(self, u, p, rho):
        soundSpeed = self.computeSoundSpeed_p_rho(p, rho)
        return np.abs(u)/soundSpeed

    def computeComprFactorZ_p_rho(self, p, rho):
        Z = FP.PropsSI('Z', 'P', p, 'D', rho, self.fluid)
        return Z

    def computeFunDerGamma_p_rho(self, p, rho):
        try:
            G = FP.PropsSI("FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS", "P", p, "D", rho, self.fluid)
            return G
        except:
            T = self.computeTemperature_p_rho(p, rho)
            try:
                Q = FP.PropsSI("Q", "T", T, "P", p, self.fluid)
            except:
                Q = 1

            G_liquid = FP.PropsSI("FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS", "T", T, "Q", 0, self.fluid)
            G_vapor = FP.PropsSI("FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS", "T", T, "Q", 1, self.fluid)

            G = (1 - Q) * G_liquid + Q * G_vapor
            return G

    def compute_gammapv_p_rho(self, p, rho):
        cp = FP.PropsSI("Cpmass", "P", p, "D", rho, self.fluid)
        cv = FP.PropsSI("Cvmass", "P", p, "D", rho, self.fluid)
        dp_drho_T = FP.PropsSI("d(P)/d(D)|T", "P", p, "D", rho, self.fluid)
        dp_dv_T = - rho**2 * dp_drho_T
        gmma_pv = -1/(p*rho) * cp/cv * dp_dv_T
        return gmma_pv

    def compute_gammapt_p_T(self, p, T):
        rho = FP.PropsSI("D", "P", p, "T", T, self.fluid)
        d_rho_dT_P = FP.PropsSI("d(D)/d(T)|P", "P", p, "T", T, self.fluid)
        dv_dT_P = - d_rho_dT_P / (rho**2)
        cp = FP.PropsSI("Cpmass", "P", p, "T", T, self.fluid)
        gamma_pT = 1 / (1 - p/cp*dv_dT_P)
        return gamma_pT

    def computeDynamicViscosity_p_rho(self, p, rho):
        # ensure inputs are numpy arrays
        p = np.asarray(p, dtype=float)
        rho = np.asarray(rho, dtype=float)
        p, rho = np.broadcast_arrays(p, rho)

        # function vectorized separately to allow for both single values and arrays 
        # and for both to still pass the phase check.
        # Vectorize the core function, passing self.fluid
        vectorized_func = np.vectorize(
            partial(self._computeDynamicViscosity_p_rho_single, fluid=self.fluid),
            otypes=[float]
        )
        return vectorized_func(p, rho)

    @staticmethod
    def _computeDynamicViscosity_p_rho_single(p: float, rho: float, fluid: str) -> float:
        # check if the state is two phase
        # readers can find interpretation of the phase number in the CoolProp documentation:
        # https://coolprop.org/_static/doxygen/html/namespace_cool_prop.html#aa1ce7c368d1058004293708038241850a648039a97f7392876038eaf56cf91e95
        # under section "phases"
        phase = FP.PropsSI("Phase", "P", p, "D", rho, fluid)

        # if phase == 6, fluid is in two-phase region.
        two_phase = False
        if phase == 6:
            two_phase = True

        def _computeDynamicViscosity_p_rho_single_phase(p: float, rho: float) -> float:
            mu = FP.PropsSI("V", "P", p, "D", rho, fluid)
            return mu

        def _computeDynamicViscosity_p_rho_two_phase(p: float, rho: float) -> float:
            y_V = FP.PropsSI("Q", "P", p, "D", rho, fluid)
            rho_V = FP.PropsSI("D", "P", p, "Q", 1, fluid)
            alpha_V = y_V * rho / rho_V
            mu_V = FP.PropsSI("V", "P", p, "Q", 1, fluid)
            mu_L = FP.PropsSI("V", "P", p, "Q", 0, fluid)
            mu_2phase = alpha_V * mu_V + (1-alpha_V) * (1+2.5*alpha_V) * mu_L
            return mu_2phase

        if not two_phase:
            mu = _computeDynamicViscosity_p_rho_single_phase(p, rho)
            return mu
        else:
            mu = _computeDynamicViscosity_p_rho_two_phase(p, rho)
            return mu

        

    def computeMach_pt_p_gammapv(self, pt, p, gamma_pv):
        """Reference to equation 8.10 Nederstigt MS thesis"""
        mach = np.sqrt(2/(gamma_pv-1) * ((pt/p)**((gamma_pv-1)/gamma_pv) - 1))
        return mach



    def computeInletQuantitiesTotal_pt_Tt(self, pressure, totPressure, totTemperature, direction):
        """The full state must be reconstructed from the quantities given in the arguments.
        The entropy of the static and total state must be the same by definition. This is used to find the temperature.
        Method used for computation of inlet static state for total state in single-phase region.

        Args:
            pressure (float): static pressure
            totPressure (float): total pressure
            totTemperature (float): total temperature
            direction (float): flow direction, integer (-1 or 1)
        """
        # compute entropy from total conditions
        entropyTotal = self.computeEntropy_p_T(totPressure, totTemperature)
        # copmpute static density from total entropy and static pressure
        density = self.computeDensity_p_S(pressure, entropyTotal)
        # compute velocity from total and static enthalpy
        enthalpyTotal = self.computeEnthalpy_p_T(totPressure, totTemperature)
        enthalpyStatic = self.computeEnthalpy_p_rho(pressure, density)
        velocity = direction * np.sqrt(2 * (enthalpyTotal - enthalpyStatic))
        # compute static energy
        energy = self.computeInternalEnergy_p_rho(pressure, density)
        return density, velocity, energy

        # # old method
        # def compute_function_residual(temperatureGuess, verbose = False):
        #     entropyStatic = self.computeEntropy_p_T(pressure, temperatureGuess)
        #     entropyTotal = self.computeEntropy_p_T(totPressure, totTemperature)
        #     residual = entropyStatic - entropyTotal
        #     if verbose:
        #         print(f"  T_guess={temperatureGuess} pressure={pressure} entropyStatic={entropyStatic} pressure={totPressure} \
        #               totTemperature={totTemperature} entropyTotal={entropyTotal} resid={residual}", flush=True)
        #         sys.stdout.flush()  
        #     return residual

        # # temperature = fsolve(compute_function_residual, totTemperature, xtol=1e-8)[0]
        # temperature, info, ier, msg = fsolve(
        #     compute_function_residual,
        #     totTemperature,
        #     xtol=1e-6,
        #     full_output=True
        # )
        # if ier != 1:
        #     raise RuntimeError(f"fsolve did not converge: {msg}")
        
        # temperature = temperature[0]
        # density = self.computeDensity_p_T(pressure, temperature)
        # gamma_pv = self.compute_gammapv_p_rho(pressure, density)
        # mach = self.computeMach_pt_p_gammapv(totPressure, pressure, gamma_pv)
        # soundSpeed = self.computeSoundSpeed_p_rho(pressure, density)
        # velocity = direction * mach * soundSpeed
        # energy = self.computeInternalEnergy_p_rho(pressure, density)
        # return density, velocity, energy

    def computeInletQuantitiesTotal_pt_Q(self, pressure, totPressure, quality, direction):
        """
        The full state must be reconstructed from the quantities given in the arguments.
        The entropy of the static and total state must be the same by definition. This is used to find the temperature.
        Method used for computation of inlet static state for total state in two-phase region.

        Args:
            pressure (float): static pressure
            totPressure (float): total pressure
            quality (float): vapor quality
            direction (float): flow direction, integer (-1 or 1)
        """
        # compute entropy from total conditions
        entropyTotal = self.computeEntropy_p_Q(totPressure, quality)
        # compute static density from total entropy and static pressure
        density = self.computeDensity_p_S(pressure, entropyTotal)
        # compute velocity from total and static enthalpy
        enthalpyTotal = self.computeEnthalpy_p_Q(totPressure, quality)
        enthalpyStatic = self.computeEnthalpy_p_rho(pressure, density)
        velocity = direction * np.sqrt(2 * (enthalpyTotal - enthalpyStatic))
        # compute static energy
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
        e = FP.PropsSI("U", "P", p, "D", rho, self.fluid)
        dp_drho_econst = FP.PropsSI("d(P)/d(D)|U", "P", p, "D", rho, self.fluid)
        dp_de_rhoconst = FP.PropsSI("d(P)/d(U)|D", "P", p, "D", rho, self.fluid)
        chi = dp_drho_econst - e/rho * dp_de_rhoconst
        kappa = dp_de_rhoconst / rho
        return chi, kappa