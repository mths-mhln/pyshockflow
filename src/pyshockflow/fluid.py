import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sys
import timeit
import fluid_properties.fluid_properties as FP

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
    def __init__(self, fluid_name, fluid_library, print_error=True):
        self.fluid_name = fluid_name
        self.fluid_library = fluid_library
        self.fluid = FP.fluid(fluid_library, fluid_name, print_error=print_error)

    def computeStaticEnergy_p_rho(self, p, rho):
        e = FP.PropsSI('U', 'P', p, 'D', rho, self.fluid)
        return e
    
    def computePressure_rho_e(self, rho, e):
        p = FP.PropsSI('P', 'D', rho, 'U', e, self.fluid)
        return p

    def computeSoundSpeed_p_rho(self, p, rho):
        try:
            a = FP.PropsSI("A", "P", p, "D", rho, self.fluid)
            return a
        except:
            # two phase region (or close) 
            T = self.computeTemperature_p_rho(p, rho)
            try:
                Q = FP.PropsSI("Q", "T", T, "P", p, self.fluid)
            except:
                # if the state is very close to saturation line it fails to find the quality -> set artifically to 1
                Q = 1

            # Speed of sound in liquid and vapor phases at the given T and P
            a_liquid = FP.PropsSI("A", "T", T, "Q", 0, self.fluid)  # sound speed for liquid phase
            a_vapor = FP.PropsSI("A", "T", T, "Q", 1, self.fluid)   # sound speed for vapor phase

            # Calculate weighted speed of sound based on quality
            a = (1 - Q) * a_liquid + Q * a_vapor
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
        """Reference to equation 8.10 Nederstigt MS thesis
        """
        mach = np.sqrt(2/(gamma_pv-1) * ((pt/p)**((gamma_pv-1)/gamma_pv) - 1))
        return mach
    

    def computeChiKappa_VinokurScheme_p_rho(self, p, rho):
        e = FP.PropsSI("U", "P", p, "D", rho, self.fluid)
        dp_drho_econst = FP.PropsSI("d(P)/d(D)|U", "P", p, "D", rho, self.fluid)
        dp_de_rhoconst = FP.PropsSI("d(P)/d(U)|D", "P", p, "D", rho, self.fluid)
        chi = dp_drho_econst - e/rho * dp_de_rhoconst
        kappa = dp_de_rhoconst / rho
        return chi, kappa
        
        
        

            