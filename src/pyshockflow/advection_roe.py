import numpy as np
from numpy import sqrt
from pyshockflow import FluidIdeal
from pyshockflow.math_utils import *


def _call_fluid_property_array(func, *args):
    """Evaluate a fluid-property function on array inputs."""
    args = [np.asarray(arg, dtype=float) for arg in args]
    return np.asarray(func(*args), dtype=float)


def _call_chi_kappa_array(fluid, p, rho):
    """Evaluate Vinokur chi/kappa arrays."""
    p = np.asarray(p, dtype=float)
    rho = np.asarray(rho, dtype=float)
    chi, kappa = fluid.computeChiKappa_VinokurScheme_p_rho(p, rho)
    return np.asarray(chi, dtype=float), np.asarray(kappa, dtype=float)


class AdvectionRoeBase:
    def __init__(self, rhoL, rhoR, uL, uR, pL, pR, fluid):
        """
        Roe scheme numerics for ideal gas. Parameters are left and right values of density, velocity and pressure, and the fluid object.
        Formulation based on x-split Riemann Solver in the book by Toro.
        """
        self.rhoL = rhoL
        self.rhoR = rhoR
        self.uL = uL
        self.uR = uR
        self.pL = pL
        self.pR = pR
        self.fluid = fluid
        if isinstance(fluid, FluidIdeal):
            self.gmma = fluid.gmma
        self.eL = fluid.computeInternalEnergy_p_rho(pL, rhoL)
        self.eR = fluid.computeInternalEnergy_p_rho(pR, rhoR)
        self.htL = self.computeTotalEnthalpy(rhoL, uL, pL, self.eL)
        self.htR = self.computeTotalEnthalpy(rhoR, uR, pR, self.eR)
        self.u1L, self.u2L, self.u3L = getConservativesFromFluidState(rhoL, uL, pL, self.fluid)
        self.u1R, self.u2R, self.u3R = getConservativesFromFluidState(rhoR, uR, pR, self.fluid)
        self.aL = self.fluid.computeSoundSpeed_p_rho(self.pL, self.rhoL)
        self.aR = self.fluid.computeSoundSpeed_p_rho(self.pR, self.rhoR)


    def computeRoeAvg(self, fL, fR):
        """
        Roe Averaging Operator
        """
        favg = (sqrt(self.rhoL)*fL + sqrt(self.rhoR)*fR)/(sqrt(self.rhoL)+ sqrt(self.rhoR))
        return favg

    
    def computeAveragedVariables(self):
        """
        compute the Roe averaged variables for the 1D Euler equations
        """
        self.rhoAVG = sqrt(self.rhoL*self.rhoR)
        self.uAVG = self.computeRoeAvg(self.uL, self.uR)
        self.hAVG = self.computeRoeAvg(self.htL, self.htR)
        self.aAVG = sqrt((self.gmma-1)*(self.hAVG-0.5*self.uAVG**2))
    
    
    def computeTotalEnthalpy(self, rho, u, p, e):
        et = 0.5*u**2 + e
        ht = et+p/rho
        return ht
    

    def computeAveragedEigenvalues(self):
        """
        compute eigenvalues of the averaged Jacobian
        """
        self.lambda_vec = np.array([self.uAVG-self.aAVG, 
                                    self.uAVG, 
                                    self.uAVG+self.aAVG])
    

    def computeAveragedEigenvectors(self):
        """
        compute eigenvector matrix of the averaged flux Jacobian
        """
        self.eigenvector_mat = np.zeros((3, 3))
        
        self.eigenvector_mat[0, 0] = 1
        self.eigenvector_mat[1, 0] = self.uAVG-self.aAVG
        self.eigenvector_mat[2, 0] = self.hAVG-self.uAVG*self.aAVG

        self.eigenvector_mat[0, 1] = 1
        self.eigenvector_mat[1, 1] = self.uAVG
        self.eigenvector_mat[2, 1] = 0.5*self.uAVG**2

        self.eigenvector_mat[0, 2] = 1
        self.eigenvector_mat[1, 2] = self.uAVG+self.aAVG
        self.eigenvector_mat[2, 2] = self.hAVG+self.uAVG*self.aAVG
    

    def computeWaveStrengths(self):
        """
        Characteristic jumps due to initial conditions
        """
        self.alphas = np.zeros(3)
        self.alphas[0] = 1/2/self.aAVG**2 *(self.pR-self.pL-self.rhoAVG*self.aAVG*(self.uR-self.uL))
        self.alphas[1] = self.rhoR-self.rhoL - (self.pR-self.pL)/self.aAVG**2
        self.alphas[2] = 1/2/self.aAVG**2*(self.pR-self.pL + self.rhoAVG*self.aAVG*(self.uR-self.uL))
        

    def computeFlux(self, entropyFixActive, fixCoefficient):
        """
        compute the Roe flux. The flux is computed for 1D problems.
        """
        self.computeAveragedVariables()
        self.computeAveragedEigenvalues()
        self.computeAveragedEigenvectors()
        self.computeWaveStrengths()
        
        fluxL = self.EulerFlux(self.u1L, self.u2L, self.u3L)
        fluxR = self.EulerFlux(self.u1R, self.u2R, self.u3R)
        fluxRoe = 0.5*(fluxL+fluxR)

        # compute the entropy fixed abs eigenvalues
        if entropyFixActive==False:
            absEig = np.abs(self.lambda_vec)
        else:
            absEig = applyEntropyFix(self.lambda_vec, self.aAVG, fixCoefficient)

        for iDim in range(3):
            for jVec in range(3):
                fluxRoe[iDim] -= 0.5*self.alphas[jVec]*absEig[jVec]*self.eigenvector_mat[iDim, jVec]
        
        return fluxRoe
        
    def EulerFlux(self, u1, u2, u3):
        """
        Get the Euler flux starting from conservative variables. 
        """
        flux1D = computeAdvectionFluxFromConservatives(u1, u2, u3, self.fluid)
        return flux1D

    @classmethod
    def computeFluxBatch(
        cls,
        rhoL,
        rhoR,
        uL,
        uR,
        pL,
        pR,
        fluid,
        entropyFixActive,
        fixCoefficient,
    ):
        """Compute Roe fluxes for all faces in one batch for the ideal-gas Roe scheme."""
        rhoL = np.asarray(rhoL, dtype=float)
        rhoR = np.asarray(rhoR, dtype=float)
        uL = np.asarray(uL, dtype=float)
        uR = np.asarray(uR, dtype=float)
        pL = np.asarray(pL, dtype=float)
        pR = np.asarray(pR, dtype=float)

        eL = _call_fluid_property_array(fluid.computeInternalEnergy_p_rho, pL, rhoL)
        eR = _call_fluid_property_array(fluid.computeInternalEnergy_p_rho, pR, rhoR)

        htL = 0.5 * uL**2 + eL + pL / rhoL
        htR = 0.5 * uR**2 + eR + pR / rhoR

        sqrtRhoL = np.sqrt(rhoL)
        sqrtRhoR = np.sqrt(rhoR)
        denom = sqrtRhoL + sqrtRhoR

        rhoAVG = np.sqrt(rhoL * rhoR)
        uAVG = (sqrtRhoL * uL + sqrtRhoR * uR) / denom
        hAVG = (sqrtRhoL * htL + sqrtRhoR * htR) / denom
        aAVG = np.sqrt((fluid.gmma - 1.0) * (hAVG - 0.5 * uAVG**2))

        eigs = np.column_stack((uAVG - aAVG, uAVG, uAVG + aAVG))
        if entropyFixActive:
            absEig = applyEntropyFix(eigs, aAVG, fixCoefficient)
        else:
            absEig = np.abs(eigs)

        deltaP = pR - pL
        deltaU = uR - uL
        a2 = aAVG**2
        alpha0 = 0.5 / a2 * (deltaP - rhoAVG * aAVG * deltaU)
        alpha1 = (rhoR - rhoL) - deltaP / a2
        alpha2 = 0.5 / a2 * (deltaP + rhoAVG * aAVG * deltaU)

        fluxL = np.column_stack((
            rhoL * uL,
            rhoL * uL**2 + pL,
            uL * (rhoL * (eL + 0.5 * uL**2) + pL),
        ))
        fluxR = np.column_stack((
            rhoR * uR,
            rhoR * uR**2 + pR,
            uR * (rhoR * (eR + 0.5 * uR**2) + pR),
        ))

        diss0 = alpha0 * absEig[:, 0] + alpha1 * absEig[:, 1] + alpha2 * absEig[:, 2]
        diss1 = (
            alpha0 * absEig[:, 0] * (uAVG - aAVG)
            + alpha1 * absEig[:, 1] * uAVG
            + alpha2 * absEig[:, 2] * (uAVG + aAVG)
        )
        diss2 = (
            alpha0 * absEig[:, 0] * (hAVG - uAVG * aAVG)
            + alpha1 * absEig[:, 1] * (0.5 * uAVG**2)
            + alpha2 * absEig[:, 2] * (hAVG + uAVG * aAVG)
        )
        diss = np.column_stack((diss0, diss1, diss2))

        return 0.5 * (fluxL + fluxR) - 0.5 * diss



class AdvectionRoeArabi(AdvectionRoeBase):
    """
    Generalised Roe Scheme for real gases, taken from the article 'A simple extension of Roe scheme for real gases', Arabi et al. 
    Journal of Computational Physics 2017. Formulation based on 1D problem.
    """
    def __init__(self, rhoL, rhoR, uL, uR, pL, pR, fluid):
        super().__init__(rhoL, rhoR, uL, uR, pL, pR, fluid)
        self.deltaP = (self.pR-self.pL)
        self.deltaU = (self.uR - self.uL)
        self.deltaRho = (self.rhoR - self.rhoL)
    
    
    def computeAveragedVariables(self):
        """
        compute the Roe averaged variables for the 1D Euler equations
        """
        self.rhoAVG = sqrt(self.rhoL*self.rhoR)
        self.uAVG = self.computeRoeAvg(self.uL, self.uR)
        self.hAVG = self.computeRoeAvg(self.htL, self.htR)
        self.aAVG = self.computeRoeAvg(self.aL, self.aR)


    def computeWaveStrengths(self):
        self.alphas = np.zeros(3)
        self.alphas[0] = 1/2/self.aAVG**2*(self.deltaP+self.rhoAVG*self.aAVG*self.deltaU)
        self.alphas[1] = 1/2/self.aAVG**2*(self.deltaP-self.rhoAVG*self.aAVG*self.deltaU)
        self.alphas[2] = self.deltaRho-self.deltaP/self.aAVG**2
    

    def computeAveragedEigenvalues(self):
        self.lambda_vec = np.array([self.uAVG+self.aAVG, 
                                    self.uAVG-self.aAVG,
                                    self.uAVG])
    

    def computeFlux(self, entropyFixActive, fixCoefficient):
        """
        Assemble the global flux, average + dissipation, following the approach of the article
        """
        self.computeAveragedVariables()
        self.computeAveragedEigenvalues()
        self.computeAveragedEigenvectors()
        self.computeWaveStrengths()
        
        fluxL = self.EulerFlux(self.u1L, self.u2L, self.u3L)
        fluxR = self.EulerFlux(self.u1R, self.u2R, self.u3R)

        # compute the entropy fixed abs eigenvalues
        if entropyFixActive==False:
            absEig = np.abs(self.lambda_vec)
        else:
            absEig = applyEntropyFix(self.lambda_vec, self.aAVG, fixCoefficient)

        deltaF = np.zeros(3)
        deltaF[0] = absEig[0]*self.alphas[0] + absEig[1]*self.alphas[1] + absEig[2]*self.alphas[2]
        deltaF[1] = (self.uAVG+self.aAVG)*absEig[0]*self.alphas[0] + (self.uAVG-self.aAVG)*absEig[1]*self.alphas[1] + self.uAVG*absEig[2]*self.alphas[2]

        X = (self.rhoR*self.uR*self.htR)-(self.rhoL*self.uL*self.htL)- \
            (self.hAVG+self.uAVG*self.aAVG)*(self.uAVG+self.aAVG)*(1/2/self.aAVG**2*(self.deltaP+self.rhoAVG*self.aAVG*self.deltaU)) - \
            (self.hAVG-self.uAVG*self.aAVG)*(self.uAVG-self.aAVG)*(1/2/self.aAVG**2*(self.deltaP-self.rhoAVG*self.aAVG*self.deltaU))
        
        if (self.uAVG>=0):
            pass
        else:
            X *= -1
        
        deltaF[2] = (self.hAVG+self.uAVG*self.aAVG)*absEig[0]*(self.alphas[0]) + \
                         (self.hAVG-self.uAVG*self.aAVG)*absEig[1]*(self.alphas[1]) + X
                         
        fluxRoe = 0.5*(fluxL+fluxR) - 0.5*deltaF
        return fluxRoe

    @classmethod
    def computeFluxBatch(
        cls,
        rhoL,
        rhoR,
        uL,
        uR,
        pL,
        pR,
        fluid,
        entropyFixActive,
        fixCoefficient,
    ):
        """Compute Roe-Arabi fluxes for all faces in one batch."""
        rhoL = np.asarray(rhoL, dtype=float)
        rhoR = np.asarray(rhoR, dtype=float)
        uL = np.asarray(uL, dtype=float)
        uR = np.asarray(uR, dtype=float)
        pL = np.asarray(pL, dtype=float)
        pR = np.asarray(pR, dtype=float)

        eL = _call_fluid_property_array(fluid.computeInternalEnergy_p_rho, pL, rhoL)
        eR = _call_fluid_property_array(fluid.computeInternalEnergy_p_rho, pR, rhoR)

        htL = 0.5 * uL**2 + eL + pL / rhoL
        htR = 0.5 * uR**2 + eR + pR / rhoR

        # Evaluate left and right sound speeds in one pass so shifted face states
        # share cache hits and the miss path is batched once.
        n_faces = rhoL.size
        p_lr = np.concatenate((pL, pR))
        rho_lr = np.concatenate((rhoL, rhoR))
        a_lr = _call_fluid_property_array(fluid.computeSoundSpeed_p_rho, p_lr, rho_lr)
        aL = a_lr[:n_faces]
        aR = a_lr[n_faces:]

        sqrtRhoL = np.sqrt(rhoL)
        sqrtRhoR = np.sqrt(rhoR)
        denom = sqrtRhoL + sqrtRhoR

        rhoAVG = np.sqrt(rhoL * rhoR)
        uAVG = (sqrtRhoL * uL + sqrtRhoR * uR) / denom
        hAVG = (sqrtRhoL * htL + sqrtRhoR * htR) / denom
        aAVG = (sqrtRhoL * aL + sqrtRhoR * aR) / denom

        deltaP = pR - pL
        deltaU = uR - uL
        deltaRho = rhoR - rhoL

        a2 = aAVG**2
        alpha0 = 0.5 / a2 * (deltaP + rhoAVG * aAVG * deltaU)
        alpha1 = 0.5 / a2 * (deltaP - rhoAVG * aAVG * deltaU)
        alpha2 = deltaRho - deltaP / a2

        eigs = np.column_stack((uAVG + aAVG, uAVG - aAVG, uAVG))
        if entropyFixActive:
            absEig = applyEntropyFix(eigs, aAVG, fixCoefficient)
        else:
            absEig = np.abs(eigs)

        fluxL = np.column_stack((
            rhoL * uL,
            rhoL * uL**2 + pL,
            uL * (rhoL * (eL + 0.5 * uL**2) + pL),
        ))
        fluxR = np.column_stack((
            rhoR * uR,
            rhoR * uR**2 + pR,
            uR * (rhoR * (eR + 0.5 * uR**2) + pR),
        ))

        deltaF = np.zeros_like(fluxL)
        deltaF[:, 0] = (
            absEig[:, 0] * alpha0
            + absEig[:, 1] * alpha1
            + absEig[:, 2] * alpha2
        )
        deltaF[:, 1] = (
            (uAVG + aAVG) * absEig[:, 0] * alpha0
            + (uAVG - aAVG) * absEig[:, 1] * alpha1
            + uAVG * absEig[:, 2] * alpha2
        )

        X = (
            (rhoR * uR * htR)
            - (rhoL * uL * htL)
            - (hAVG + uAVG * aAVG) * (uAVG + aAVG) * (0.5 / a2 * (deltaP + rhoAVG * aAVG * deltaU))
            - (hAVG - uAVG * aAVG) * (uAVG - aAVG) * (0.5 / a2 * (deltaP - rhoAVG * aAVG * deltaU))
        )
        X = np.where(uAVG >= 0.0, X, -X)
        deltaF[:, 2] = (
            (hAVG + uAVG * aAVG) * absEig[:, 0] * alpha0
            + (hAVG - uAVG * aAVG) * absEig[:, 1] * alpha1
            + X
        )

        return 0.5 * (fluxL + fluxR) - 0.5 * deltaF
    
    

class AdvectionRoeVinokur(AdvectionRoeBase):
    """
    Generalised Roe Scheme for real gases, 
    where the Roe avg state is taken from the article 'Generalized flux-vector splitting and Roe average for an equilibrium real gas', Vinokur and Montagnè 
    Journal of Computational Physics 1990. Formulation based on 1D problem.
    """
    def __init__(self, rhoL, rhoR, uL, uR, pL, pR, fluid):
        super().__init__(rhoL, rhoR, uL, uR, pL, pR, fluid)
        self.deltaP = (self.pR-self.pL)
        self.deltaU = (self.uR - self.uL)
        self.deltaRho = (self.rhoR - self.rhoL)
    
    
    def computeAveragedVariables(self):
        """
        compute the Roe averaged state following the approach described in the articleof Vinokur
        """
        alpha = np.sqrt(self.rhoL) / (np.sqrt(self.rhoL)+np.sqrt(self.rhoR))
        self.uAVG = alpha*self.uL + (1-alpha)*self.uR
        self.htAVG = alpha*self.htL + (1-alpha)*self.htR
        self.hL = self.htL - 0.5*self.uL**2
        self.hR = self.htR - 0.5*self.uR**2
        self.hAVG = alpha*self.hL + (1-alpha)*self.hR + 0.5*alpha*(1.0-alpha)*self.deltaU**2
        
        # compute mean initial guess state
        p_mean = 0.5*(self.pL+self.pR)
        eL = self.fluid.computeInternalEnergy_p_rho(self.pL, self.rhoL)
        rho_mean = 0.5*(self.rhoL+self.rhoR)
        rhoeL = self.rhoL*eL
        eR = self.fluid.computeInternalEnergy_p_rho(self.pR, self.rhoR)
        rhoeR = self.rhoR*eR
        rhoe_mean = 0.5*(rhoeL+rhoeR)
        e_mean = rhoe_mean/rho_mean
        
        chiL, kappaL = self.fluid.computeChiKappa_VinokurScheme_p_rho(self.pL, self.rhoL)
        chiR, kappaR = self.fluid.computeChiKappa_VinokurScheme_p_rho(self.pR, self.rhoR)
        chiM, kappaM = self.fluid.computeChiKappa_VinokurScheme_p_rho(p_mean, rho_mean)
        chiHat = (chiL + chiR + 4.0*chiM) / 6.0
        kappaHat = (kappaL + kappaR + 4.0*kappaM) / 6.0
        delta_rhoe = (rhoeR - rhoeL)
        
        # projection procedure to compute the average state starting fro the initial guess (hat values)
        error_term = self.deltaP - chiHat*self.deltaRho - kappaHat*delta_rhoe
        hM = 0.5*(self.hL+self.hR)
        kappah_hat = (kappaL*self.hL + kappaR*self.hR + 4.0*kappaM*hM) / 6.0
        csquare_L = chiL + kappaL*self.hL
        csquare_R = chiR + kappaR*self.hR
        csquare_M = chiM + kappaM*hM
        sHat = (csquare_L + csquare_R + 4.0*csquare_M) / 6.0
        D_term = (sHat*self.deltaRho)**2 + (self.deltaP)**2
        if self.deltaRho==0:
            self.chiAVG = chiHat
        else:
            self.chiAVG = (D_term * chiHat + sHat**2 * self.deltaRho * error_term) / (D_term - self.deltaP*error_term)
        
        if self.deltaP==0:
            self.kappaAVG = kappaHat
        else:
            self.kappaAVG = (D_term * kappaHat) / (D_term - self.deltaP*error_term)
        
        self.aAVG = np.sqrt(self.chiAVG + self.kappaAVG*self.hAVG)
    
    
    def computeFlux(self, entropyFixActive, fixCoefficient):
        """
        compute the global flux, average + dissipation
        """
        fluxL = self.EulerFlux(self.u1L, self.u2L, self.u3L)
        fluxR = self.EulerFlux(self.u1R, self.u2R, self.u3R)

        # compute the Eigenvectors matrices
        k1 = 0.5*self.kappaAVG*self.uAVG**2 + self.kappaAVG
        k2 = 0.5*self.uAVG**2 - self.chiAVG/self.kappaAVG
        
        # right eigenvectors matrix
        matrixR = np.array([[1, 1, 1],
                            [self.uAVG, self.uAVG+self.aAVG, self.uAVG-self.aAVG],
                            [k2, self.htAVG + self.aAVG*self.uAVG, self.htAVG - self.aAVG*self.uAVG]])
        
        # left eigenvectors matrix
        matrixRinv = np.array([[1-k1/self.aAVG**2, self.kappaAVG*self.uAVG/self.aAVG**2, -self.kappaAVG/self.aAVG**2],
                               [0.5*(k1/self.aAVG**2-self.uAVG/self.aAVG), -0.5*(self.kappaAVG*self.uAVG/self.aAVG**2-1/self.aAVG), 0.5*self.kappaAVG/self.aAVG**2],
                               [0.5*(k1/self.aAVG**2+self.uAVG/self.aAVG), -0.5*(self.kappaAVG*self.uAVG/self.aAVG**2+1/self.aAVG), 0.5*self.kappaAVG/self.aAVG**2]])
                
        # eigenvalues, to fix
        eigsAVG = np.array([self.uAVG, self.uAVG+self.aAVG, self.uAVG-self.aAVG])
        if entropyFixActive==False:
            absEig = np.abs(eigsAVG)
        else:
            absEig = applyEntropyFix(eigsAVG, self.aAVG, fixCoefficient)
        
        # eigenvalues matrix
        matrixLambda = np.diag(absEig)
        
        # compute the Flux
        deltaU = np.array([self.u1R-self.u1L, self.u2R-self.u2L, self.u3R-self.u3L]).reshape(3,1)
        deltaFlux = matrixR @ matrixLambda @ matrixRinv @ deltaU
        fluxRoe = 0.5*(fluxL+fluxR) - 0.5*deltaFlux.flatten()
        return fluxRoe

    @classmethod
    def computeFluxBatch(
        cls,
        rhoL,
        rhoR,
        uL,
        uR,
        pL,
        pR,
        fluid,
        entropyFixActive,
        fixCoefficient,
    ):
        """Compute Roe-Vinokur fluxes for all faces in one batch."""
        rhoL = np.asarray(rhoL, dtype=float)
        rhoR = np.asarray(rhoR, dtype=float)
        uL = np.asarray(uL, dtype=float)
        uR = np.asarray(uR, dtype=float)
        pL = np.asarray(pL, dtype=float)
        pR = np.asarray(pR, dtype=float)

        eL = _call_fluid_property_array(fluid.computeInternalEnergy_p_rho, pL, rhoL)
        eR = _call_fluid_property_array(fluid.computeInternalEnergy_p_rho, pR, rhoR)

        htL = 0.5 * uL**2 + eL + pL / rhoL
        htR = 0.5 * uR**2 + eR + pR / rhoR

        deltaP = pR - pL
        deltaU = uR - uL
        deltaRho = rhoR - rhoL

        sqrtRhoL = np.sqrt(rhoL)
        sqrtRhoR = np.sqrt(rhoR)
        alpha = sqrtRhoL / (sqrtRhoL + sqrtRhoR)

        uAVG = alpha * uL + (1.0 - alpha) * uR
        htAVG = alpha * htL + (1.0 - alpha) * htR
        hL = htL - 0.5 * uL**2
        hR = htR - 0.5 * uR**2
        hAVG = alpha * hL + (1.0 - alpha) * hR + 0.5 * alpha * (1.0 - alpha) * deltaU**2

        p_mean = 0.5 * (pL + pR)
        rho_mean = 0.5 * (rhoL + rhoR)
        rhoeL = rhoL * eL
        rhoeR = rhoR * eR

        chiL, kappaL = _call_chi_kappa_array(fluid, pL, rhoL)
        chiR, kappaR = _call_chi_kappa_array(fluid, pR, rhoR)
        chiM, kappaM = _call_chi_kappa_array(fluid, p_mean, rho_mean)

        chiHat = (chiL + chiR + 4.0 * chiM) / 6.0
        kappaHat = (kappaL + kappaR + 4.0 * kappaM) / 6.0
        delta_rhoe = rhoeR - rhoeL

        error_term = deltaP - chiHat * deltaRho - kappaHat * delta_rhoe
        hM = 0.5 * (hL + hR)
        csquare_L = chiL + kappaL * hL
        csquare_R = chiR + kappaR * hR
        csquare_M = chiM + kappaM * hM
        sHat = (csquare_L + csquare_R + 4.0 * csquare_M) / 6.0
        D_term = (sHat * deltaRho) ** 2 + deltaP**2

        denom = D_term - deltaP * error_term
        chiAVG = np.where(
            deltaRho == 0.0,
            chiHat,
            (D_term * chiHat + sHat**2 * deltaRho * error_term) / denom,
        )
        kappaAVG = np.where(
            deltaP == 0.0,
            kappaHat,
            (D_term * kappaHat) / denom,
        )
        aAVG = np.sqrt(chiAVG + kappaAVG * hAVG)

        fluxL = np.column_stack((
            rhoL * uL,
            rhoL * uL**2 + pL,
            uL * (rhoL * (eL + 0.5 * uL**2) + pL),
        ))
        fluxR = np.column_stack((
            rhoR * uR,
            rhoR * uR**2 + pR,
            uR * (rhoR * (eR + 0.5 * uR**2) + pR),
        ))

        k1 = 0.5 * kappaAVG * uAVG**2 + kappaAVG
        k2 = 0.5 * uAVG**2 - chiAVG / kappaAVG

        matrixR = np.zeros((rhoL.size, 3, 3), dtype=float)
        matrixR[:, 0, 0] = 1.0
        matrixR[:, 0, 1] = 1.0
        matrixR[:, 0, 2] = 1.0
        matrixR[:, 1, 0] = uAVG
        matrixR[:, 1, 1] = uAVG + aAVG
        matrixR[:, 1, 2] = uAVG - aAVG
        matrixR[:, 2, 0] = k2
        matrixR[:, 2, 1] = htAVG + aAVG * uAVG
        matrixR[:, 2, 2] = htAVG - aAVG * uAVG

        matrixRinv = np.zeros((rhoL.size, 3, 3), dtype=float)
        matrixRinv[:, 0, 0] = 1.0 - k1 / aAVG**2
        matrixRinv[:, 0, 1] = kappaAVG * uAVG / aAVG**2
        matrixRinv[:, 0, 2] = -kappaAVG / aAVG**2
        matrixRinv[:, 1, 0] = 0.5 * (k1 / aAVG**2 - uAVG / aAVG)
        matrixRinv[:, 1, 1] = -0.5 * (kappaAVG * uAVG / aAVG**2 - 1.0 / aAVG)
        matrixRinv[:, 1, 2] = 0.5 * kappaAVG / aAVG**2
        matrixRinv[:, 2, 0] = 0.5 * (k1 / aAVG**2 + uAVG / aAVG)
        matrixRinv[:, 2, 1] = -0.5 * (kappaAVG * uAVG / aAVG**2 + 1.0 / aAVG)
        matrixRinv[:, 2, 2] = 0.5 * kappaAVG / aAVG**2

        eigs = np.column_stack((uAVG, uAVG + aAVG, uAVG - aAVG))
        if entropyFixActive:
            absEig = applyEntropyFix(eigs, aAVG, fixCoefficient)
        else:
            absEig = np.abs(eigs)

        u1L = rhoL
        u2L = rhoL * uL
        u3L = rhoL * (0.5 * uL**2 + eL)
        u1R = rhoR
        u2R = rhoR * uR
        u3R = rhoR * (0.5 * uR**2 + eR)
        deltaCons = np.column_stack((u1R - u1L, u2R - u2L, u3R - u3L))

        projected = np.einsum("nij,nj->ni", matrixRinv, deltaCons)
        projected *= absEig
        deltaFlux = np.einsum("nij,nj->ni", matrixR, projected)

        return 0.5 * (fluxL + fluxR) - 0.5 * deltaFlux
        

def applyEntropyFix(eigs, aAVG, kappa):
    """
    Apply Harten entropy fix to eigenvalues.
    
    eigs : ndarray of shape (3,)
        Raw Roe eigenvalues [u, u+a, u-a].
    aAVG : float
        Roe-averaged sound speed.
    kappa : float
        Fix coefficient (default 0.2).
    """
    eigs = np.asarray(eigs, dtype=float)
    delta = kappa * np.asarray(aAVG, dtype=float)

    if eigs.ndim == 1:
        delta_eff = np.maximum(delta, 1e-14)
    else:
        delta_eff = np.maximum(delta[..., None], 1e-14)

    abs_eigs = np.abs(eigs)
    fixed_small = 0.5 * (eigs**2 / delta_eff + delta_eff)
    return np.where(abs_eigs < delta_eff, fixed_small, abs_eigs)
















