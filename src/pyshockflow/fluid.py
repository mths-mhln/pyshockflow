import sys

import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math
import fluid_properties.fluid_properties as FP
from functools import partial

class FluidIdeal():
    """
    Ideal Fluid Class, where thermodynamic properties and transformation are computed with ideal gas laws
    """
    def __init__(self, gmma, Rgas):
        self.gmma = gmma
        self.Rgas = Rgas
    
    def computeStaticEnergy_p_rho(self, p, rho):
        return (p / (self.gmma - 1) / rho)
    
    def computePressure_rho_e(self, rho, e):
        return (self.gmma-1)*rho*e
    
    def computeSoundSpeed_p_rho(self, p, rho):
        return np.sqrt(self.gmma*p/rho)
    
    def computeMach_u_p_rho(self, u, p, rho):
        soundSpeed = self.computeSoundSpeed_p_rho(p, rho)
        return np.abs(u)/soundSpeed

    def computeTemperature_p_rho(self, p, rho):
        return (p/rho)/self.Rgas

    def computeDensity_p_T(self, p, T):
        return p/self.Rgas/T

    def computeEntropy_p_rho(self, p, rho):
        return p/(rho**self.gmma)

    def computeFunDerGamma_p_rho(self, p, rho):
        if isinstance(p, np.ndarray): # handle the case when the inputs are arrays
            return 0.5*(self.gmma+1)+np.zeros_like(p)
        else:
            return 0.5*(self.gmma+1)

    def computeComprFactorZ_p_rho(self, p, rho):
        if isinstance(p, np.ndarray):
            return 1+np.zeros_like(p)
        else:
            return 1

    def computeTotalPressure_p_M(self, p, M):
        return p*(1+(self.gmma-1)/2*M**2)**(self.gmma/(self.gmma-1))

    def computeMach_pt_p(self, pt, p):
        mach = np.sqrt( 2/(self.gmma-1) * ((pt/p)**((self.gmma-1)/self.gmma)-1) )
        return mach

    def computeTotalTemperature_T_M(self, T, M):
        return T*(1+(self.gmma-1)/2*M**2)

    def computeTemperature_Tt_M(self, Tt, M):
        return Tt/(1+(self.gmma-1)/2*M**2)

    def computePressure_Pt_M(self, Pt, M):
        return Pt/((1+(self.gmma-1)/2*M**2)**(self.gmma/(self.gmma-1)))
    
    def computeInletQuantitiesTotal(self, pressure, totPressure, totTemperature, direction):
        mach = self.computeMach_pt_p(totPressure, pressure)
        temperature = self.computeTemperature_Tt_M(totTemperature, mach)
        density = self.computeDensity_p_T(pressure, temperature)
        soundSpeed = self.computeSoundSpeed_p_rho(pressure, density)
        velocity = mach*soundSpeed*direction
        energy = self.computeStaticEnergy_p_rho(pressure, density)
        return density, velocity, energy
    
    def compute_gammapv_p_rho(self, p, rho):
        if isinstance(p, np.ndarray):
            gmma_pv = np.zeros_like(p)+self.gmma
        else:
            gmma_pv = self.gmma
        return gmma_pv

    def computeChiKappa_VinokurScheme_p_rho(self, p, rho):
        chi = 0
        kappa = self.gmma-1
        return chi, kappa


class FluidReal():
    """
    Real Fluid Class, where thermodynamic properties and transformations are taken from coolprop
    """
    def __init__(self, fluid_name, fluid_library, property_extraction_method, print_error=True):
        self.fluid_name = fluid_name
        self.fluid_library = fluid_library
        if property_extraction_method.lower() == 'fluid':
            self.fluid = FP.fluid(fluid_library, fluid_name,  print_error=print_error)
        if property_extraction_method.lower() == 'abstractstate':
            self.fluid = FP.AbstractState(fluid_library, fluid_name)
        if property_extraction_method.lower() == 'abstractstate_v2':
            self.fluid = FP.AbstractState_v2(fluid_library, fluid_name)
            

    def computeStaticEnergy_p_rho(self, p, rho):
        e = FP.PropsSI('U', 'P', p, 'D', rho, self.fluid)
        return e
    
    def computePressure_rho_e(self, rho, e):
        p = FP.PropsSI('P', 'D', rho, 'U', e, self.fluid)
        return p
    
    def computeSoundSpeed_p_rho(self, p, rho):
        """Public method - handles self.fluid and vectorization"""
        # Ensure inputs are numpy arrays
        p = np.asarray(p, dtype=float)
        rho = np.asarray(rho, dtype=float)
        p, rho = np.broadcast_arrays(p, rho)
        
        # Vectorize the core function, passing self.fluid
        vectorized_func = np.vectorize(
            partial(self._computeSoundSpeed_p_rho_single, fluid=self.fluid),
            otypes=[float]
        )
        
        return vectorized_func(p, rho)

    @staticmethod
    def _computeSoundSpeed_p_rho_single(p, rho, fluid):
        """Core scalar function - no self, pure computation"""
        # check if the state is single phase or two phase
        T = FP.PropsSI("T", "P", p, "D", rho, fluid)
        T_crit = FP.PropsSI("Tcrit", fluid_object = fluid)
        if T < 0.99 * T_crit: 
            # 0.99 because an evaluation had T = 304.1281982111877, T_crit = 304.1282 (CO2) 
            # and S_sat_V was not defined, which is acceptable from CoolProp
            S_sat_V = FP.PropsSI("S", "T", T, "Q", 1, fluid)
            S_sat_L = FP.PropsSI("S", "T", T, "Q", 0, fluid)
            non_saturable = False
        else:
            non_saturable = True
        S = FP.PropsSI("S", "P", p, "D", rho, fluid)

        def _computeSoundSpeed_p_rho_single_phase(p, rho, fluid):
            a = FP.PropsSI("A", "P", p, "D", rho, fluid)
            return a
        
        def _computeSoundSpeed_p_rho_two_phase(p, rho, fluid):
            # two-phase (HEM model from Cioffi et al.)
            x_V = FP.PropsSI("Q", "P", p, "D", rho, fluid)
            x_L = 1 - x_V
            soundSpeed_L = FP.PropsSI("A", "P", p, "Q", 0, fluid)
            soundSpeed_V = FP.PropsSI("A", "P", p, "Q", 1, fluid)
            rho_L = FP.PropsSI("D", "P", p, "Q", 0, fluid)
            rho_V = FP.PropsSI("D", "P", p, "Q", 1, fluid)
            c_p_L = FP.PropsSI("Cpmass", "P", p, "Q", 0, fluid)
            c_p_V = FP.PropsSI("Cpmass", "P", p, "Q", 1, fluid)
            alpha_V = x_V * (rho/rho_V)
            alpha_L = x_L * (rho/rho_L)
            
            # Finite difference for ds/dp at constant Q
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
        
        if non_saturable:
            # only option is single phase
            a = _computeSoundSpeed_p_rho_single_phase(p, rho, fluid)
            return a
        else:
            # can be two-phase or single phase:
            if S <= S_sat_L or S >= S_sat_V:
                # try single phase first. At boundary can yield some errors.
                try: 
                    a = _computeSoundSpeed_p_rho_single_phase(p, rho, fluid)
                except:
                    a = _computeSoundSpeed_p_rho_two_phase(p, rho, fluid)
                return a
            else:
                a = _computeSoundSpeed_p_rho_two_phase(p, rho, fluid)
                return a 
        

    def computeMach_u_p_rho(self, u, p, rho):
        soundSpeed = self.computeSoundSpeed_p_rho(p, rho)
        return np.abs(u)/soundSpeed

    def computeTemperature_p_rho(self, p, rho):
        T = FP.PropsSI('T', 'P', p, 'D', rho, self.fluid)
        return T

    def computeDensity_p_T(self, p, T):
        rho = FP.PropsSI('D', 'P', p, 'T', T, self.fluid)
        return rho

    def computeEntropy_p_rho(self, p, rho):
        s = FP.PropsSI('S', 'P', p, 'D', rho, self.fluid)
        return s

    def computeEntropy_p_T(self, p, T):
        s = FP.PropsSI('S', 'P', p, 'T', T, self.fluid)
        return s

    def computeFunDerGamma_p_rho(self, p, rho):
        try: # if single phase this will work
            G = FP.PropsSI("FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS", "P", p, "D", rho, self.fluid)
            return G
        except: # if close to two phase, we need to do like the speed of sound
            T = self.computeTemperature_p_rho(p, rho)
            try:
                Q = FP.PropsSI("Q", "T", T, "P", p, self.fluid)
            except:
                # if the state is very close to saturation line it fails to find the quality -> set artifically to 1
                Q = 1

            # G in liquid and vapor phases at the given T
            G_liquid = FP.PropsSI("FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS", "T", T, "Q", 0, self.fluid)  # sound speed for liquid phase
            G_vapor = FP.PropsSI("FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS", "T", T, "Q", 1, self.fluid)   # sound speed for vapor phase

            # Calculate weighted G based on quality
            G = (1 - Q) * G_liquid + Q * G_vapor
            return G

    def computeComprFactorZ_p_rho(self, p, rho):
        Z = FP.PropsSI('Z', 'P', p, 'D', rho, self.fluid)
        return Z

    
    def computeInletQuantitiesTotal(self, pressure, totPressure, totTemperature, direction):
        """The full state must be reconstructed from the quantities given in the arguments.
        The entropy of the static and total state must be the same by definition. This is used to find the temperature.

        Args:
            pressure (float): static pressure
            totPressure (float): total pressure
            totTemperature (float): total temperature
        """
        def compute_function_residual(temperatureGuess):
            entropyStatic = self.computeEntropy_p_T(pressure, temperatureGuess)
            entropyTotal = self.computeEntropy_p_T(totPressure, totTemperature)
            residual = entropyStatic - entropyTotal
            print(f"  T_guess={temperatureGuess} pressure={pressure} entropyStatic={entropyStatic} pressure={totPressure} totTemperature={totTemperature} entropyTotal={entropyTotal} resid={residual}", flush=True)
            sys.stdout.flush()  # belt-and-suspenders
            return residual

        # temperature = fsolve(compute_function_residual, totTemperature, xtol=1e-8)[0]
        temperature, info, ier, msg = fsolve(
            compute_function_residual,
            totTemperature,
            xtol=1e-6,
            full_output=True
        )
        if ier != 1:
            raise RuntimeError(f"fsolve did not converge: {msg}")
        
        temperature = temperature[0]
        density = self.computeDensity_p_T(pressure, temperature)
        gamma_pv = self.compute_gammapv_p_rho(pressure, density)
        mach = self.computeMach_pt_p_gammapv(totPressure, pressure, gamma_pv)
        soundSpeed = self.computeSoundSpeed_p_rho(pressure, density)
        velocity = direction * mach * soundSpeed
        energy = self.computeStaticEnergy_p_rho(pressure, density)
        return density, velocity, energy

    def computeInletQuantitiesStatic(self, pressure, enthalpy):
        density = self.computeDensity_p_h(pressure, enthalpy)
        energy = self.computeStaticEnergy_p_rho(pressure, density)
        return density, energy
    
    def computeDensity_p_h(self, p, h):
        return FP.PropsSI('D', 'P', p, 'H', h, self.fluid)
    
    def computeEnthalpy_p_T(self, p, T):
        return FP.PropsSI('H', 'P', p, 'T', T, self.fluid)

    def computeEnthalpy_p_s(self, p, s):
        return FP.PropsSI('H', 'P', p, 'S', s, self.fluid)
    
    def computeDensity_p_s(self, p, s):
        return FP.PropsSI('D', 'P', p, 'S', s, self.fluid)

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


    def computeMach_pt_p_gammapv(self, pt, p, gamma_pv):
        """Reference to equation 8.10 Nederstigt MS thesis"""
        mach = np.sqrt(2/(gamma_pv-1) * ((pt/p)**((gamma_pv-1)/gamma_pv) - 1))
        return mach
    

    def computeChiKappa_VinokurScheme_p_rho(self, p, rho):
        e = FP.PropsSI("U", "P", p, "D", rho, self.fluid)
        dp_drho_econst = FP.PropsSI("d(P)/d(D)|U", "P", p, "D", rho, self.fluid)
        dp_de_rhoconst = FP.PropsSI("d(P)/d(U)|D", "P", p, "D", rho, self.fluid)
        chi = dp_drho_econst - e/rho * dp_de_rhoconst
        kappa = dp_de_rhoconst / rho
        return chi, kappa
        
        
        

            