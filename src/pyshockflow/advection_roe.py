import numpy as np
from numpy import sqrt
from pyshockflow import FluidIdeal
from pyshockflow.math_utils import *




class AdvectionRoeBase:
    def __init__(self, rhoL, rhoR, uL, uR, pL, pR, fluid):
        """
        Roe scheme numerics for ideal gas. Parameters are left and right values of
        density, velocity and pressure, and the fluid object. 
        Formulation based on x-split Riemann Solver in the book by Toro.
        """
        self.rhoL = rhoL
        self.rhoR = rhoR
        self.uL = uL
        self.uR = uR
        self.pL = pL
        self.pR = pR
        self.fluid = fluid
        self.nFaces = self.rhoL.size

        if isinstance(fluid, FluidIdeal):
            self.gmma = fluid.gmma

        self.eL = fluid.computeInternalEnergy_p_rho(self.pL, self.rhoL)
        self.eR = fluid.computeInternalEnergy_p_rho(self.pR, self.rhoR)

        self.htL = self.computeTotalEnthalpy(self.rhoL, self.uL, self.pL, self.eL)
        self.htR = self.computeTotalEnthalpy(self.rhoR, self.uR, self.pR, self.eR)

        # Conservative variables, built directly from primitives (u3 = rho*(e + 0.5 u^2)
        # by definition, independent of the EOS), matching the batched implementation.
        self.u1L, self.u2L, self.u3L = self.rhoL, self.rhoL * self.uL, self.rhoL * (self.eL + 0.5 * self.uL**2)
        self.u1R, self.u2R, self.u3R = self.rhoR, self.rhoR * self.uR, self.rhoR * (self.eR + 0.5 * self.uR**2)

        self.aL = fluid.computeSoundSpeed_p_rho(self.pL, self.rhoL)
        self.aR = fluid.computeSoundSpeed_p_rho(self.pR, self.rhoR)

    def computeRoeAvg(self, fL, fR):
        """
        Roe Averaging Operator
        """
        favg = (sqrt(self.rhoL) * fL + sqrt(self.rhoR) * fR) / (sqrt(self.rhoL) + sqrt(self.rhoR))
        return favg

    def computeAveragedVariables(self):
        """
        compute the Roe averaged variables for the 1D Euler equations
        """
        self.rhoAVG = sqrt(self.rhoL * self.rhoR)
        self.uAVG = self.computeRoeAvg(self.uL, self.uR)
        self.hAVG = self.computeRoeAvg(self.htL, self.htR)
        self.aAVG = sqrt((self.gmma - 1) * (self.hAVG - 0.5 * self.uAVG**2))

    def computeTotalEnthalpy(self, rho, u, p, e):
        et = 0.5 * u**2 + e
        ht = et + p / rho
        return ht

    def computeAveragedEigenvalues(self):
        """
        compute eigenvalues of the averaged Jacobian. Shape (nFaces, 3).
        """
        self.lambda_vec = np.column_stack((self.uAVG - self.aAVG,
                                            self.uAVG,
                                            self.uAVG + self.aAVG))

    def computeAveragedEigenvectors(self):
        """
        compute eigenvector matrix of the averaged flux Jacobian. Shape (nFaces, 3, 3).
        """
        self.eigenvector_mat = np.zeros((self.nFaces, 3, 3))

        self.eigenvector_mat[:, 0, 0] = 1
        self.eigenvector_mat[:, 1, 0] = self.uAVG - self.aAVG
        self.eigenvector_mat[:, 2, 0] = self.hAVG - self.uAVG * self.aAVG

        self.eigenvector_mat[:, 0, 1] = 1
        self.eigenvector_mat[:, 1, 1] = self.uAVG
        self.eigenvector_mat[:, 2, 1] = 0.5 * self.uAVG**2

        self.eigenvector_mat[:, 0, 2] = 1
        self.eigenvector_mat[:, 1, 2] = self.uAVG + self.aAVG
        self.eigenvector_mat[:, 2, 2] = self.hAVG + self.uAVG * self.aAVG

    def computeWaveStrengths(self):
        """
        Characteristic jumps due to initial conditions. Shape (nFaces, 3).
        """
        self.alphas = np.zeros((self.nFaces, 3))
        self.alphas[:, 0] = 0.5 / self.aAVG**2 * (self.pR - self.pL - self.rhoAVG * self.aAVG * (self.uR - self.uL))
        self.alphas[:, 1] = (self.rhoR - self.rhoL) - (self.pR - self.pL) / self.aAVG**2
        self.alphas[:, 2] = 0.5 / self.aAVG**2 * (self.pR - self.pL + self.rhoAVG * self.aAVG * (self.uR - self.uL))

    def _assembleFlux(self, rho, u, p, e):
        """Euler flux from primitives, shape (nFaces, 3)."""
        return np.column_stack((rho * u,
                                 rho * u**2 + p,
                                 u * (rho * (e + 0.5 * u**2) + p)))

    def computeFlux(self, entropyFixActive, fixCoefficient):
        """
        compute the Roe flux for all faces. Returns shape (nFaces, 3).
        """
        self.computeAveragedVariables()
        self.computeAveragedEigenvalues()
        self.computeAveragedEigenvectors()
        self.computeWaveStrengths()

        fluxL = self._assembleFlux(self.rhoL, self.uL, self.pL, self.eL)
        fluxR = self._assembleFlux(self.rhoR, self.uR, self.pR, self.eR)
        fluxRoe = 0.5 * (fluxL + fluxR)

        if entropyFixActive == False:
            absEig = np.abs(self.lambda_vec)
        else:
            absEig = applyEntropyFix(self.lambda_vec, self.aAVG, fixCoefficient)

        # dissipation[n, i] = sum_j alphas[n, j] * absEig[n, j] * eigenvector_mat[n, i, j]
        dissipation = np.einsum('nj,nj,nij->ni', self.alphas, absEig, self.eigenvector_mat)
        fluxRoe = fluxRoe - 0.5 * dissipation

        return fluxRoe

    def EulerFlux(self, u1, u2, u3):
        """
        Get the Euler flux starting from conservative variables.
        """
        flux1D = computeAdvectionFluxFromConservatives(u1, u2, u3, self.fluid)
        return flux1D
    


class AdvectionRoeArabi(AdvectionRoeBase):
    """
    Generalised Roe Scheme for real gases, taken from the article 'A simple extension
    of Roe scheme for real gases', Arabi et al. Journal of Computational Physics 2017.
    """
    def __init__(self, rhoL, rhoR, uL, uR, pL, pR, fluid):
        super().__init__(rhoL, rhoR, uL, uR, pL, pR, fluid)
        self.deltaP = self.pR - self.pL
        self.deltaU = self.uR - self.uL
        self.deltaRho = self.rhoR - self.rhoL

    def computeAveragedVariables(self):
        """
        compute the Roe averaged variables for the 1D Euler equations
        """
        self.rhoAVG = sqrt(self.rhoL * self.rhoR)
        self.uAVG = self.computeRoeAvg(self.uL, self.uR)
        self.hAVG = self.computeRoeAvg(self.htL, self.htR)
        self.aAVG = self.computeRoeAvg(self.aL, self.aR)

    def computeWaveStrengths(self):
        self.alphas = np.zeros((self.nFaces, 3))
        self.alphas[:, 0] = 0.5 / self.aAVG**2 * (self.deltaP + self.rhoAVG * self.aAVG * self.deltaU)
        self.alphas[:, 1] = 0.5 / self.aAVG**2 * (self.deltaP - self.rhoAVG * self.aAVG * self.deltaU)
        self.alphas[:, 2] = self.deltaRho - self.deltaP / self.aAVG**2

    def computeAveragedEigenvalues(self):
        self.lambda_vec = np.column_stack((self.uAVG + self.aAVG,
                                            self.uAVG - self.aAVG,
                                            self.uAVG))

    def computeFlux(self, entropyFixActive, fixCoefficient):
        """
        Assemble the global flux, average + dissipation, following the approach of the article.
        Returns shape (nFaces, 3), or (3,) for scalar input.
        """
        self.computeAveragedVariables()
        self.computeAveragedEigenvalues()
        self.computeAveragedEigenvectors()
        self.computeWaveStrengths()

        fluxL = self._assembleFlux(self.rhoL, self.uL, self.pL, self.eL)
        fluxR = self._assembleFlux(self.rhoR, self.uR, self.pR, self.eR)

        if entropyFixActive == False:
            absEig = np.abs(self.lambda_vec)
        else:
            absEig = applyEntropyFix(self.lambda_vec, self.aAVG, fixCoefficient)

        eig0, eig1, eig2 = absEig[:, 0], absEig[:, 1], absEig[:, 2]
        alpha0, alpha1, alpha2 = self.alphas[:, 0], self.alphas[:, 1], self.alphas[:, 2]

        deltaF0 = eig0 * alpha0 + eig1 * alpha1 + eig2 * alpha2
        deltaF1 = ((self.uAVG + self.aAVG) * eig0 * alpha0
                   + (self.uAVG - self.aAVG) * eig1 * alpha1
                   + self.uAVG * eig2 * alpha2)

        X = (self.rhoR * self.uR * self.htR) - (self.rhoL * self.uL * self.htL) \
            - (self.hAVG + self.uAVG * self.aAVG) * (self.uAVG + self.aAVG) * alpha0 \
            - (self.hAVG - self.uAVG * self.aAVG) * (self.uAVG - self.aAVG) * alpha1
        X = np.where(self.uAVG >= 0, X, -X)

        deltaF2 = (self.hAVG + self.uAVG * self.aAVG) * eig0 * alpha0 \
                  + (self.hAVG - self.uAVG * self.aAVG) * eig1 * alpha1 \
                  + X

        deltaF = np.column_stack((deltaF0, deltaF1, deltaF2))
        fluxRoe = 0.5 * (fluxL + fluxR) - 0.5 * deltaF

        return fluxRoe
    


class AdvectionRoeVinokur(AdvectionRoeBase):
    """
    Generalised Roe Scheme for real gases, where the Roe avg state is taken from the
    article 'Generalized flux-vector splitting and Roe average for an equilibrium real
    gas', Vinokur and Montagnè, Journal of Computational Physics 1990.
    """
    def __init__(self, rhoL, rhoR, uL, uR, pL, pR, fluid):
        super().__init__(rhoL, rhoR, uL, uR, pL, pR, fluid)
        self.deltaP = self.pR - self.pL
        self.deltaU = self.uR - self.uL
        self.deltaRho = self.rhoR - self.rhoL

    def computeAveragedVariables(self):
        """
        compute the Roe averaged state following the approach described in the
        article of Vinokur
        """
        alpha = np.sqrt(self.rhoL) / (np.sqrt(self.rhoL) + np.sqrt(self.rhoR))
        self.uAVG = alpha * self.uL + (1 - alpha) * self.uR
        self.htAVG = alpha * self.htL + (1 - alpha) * self.htR
        self.hL = self.htL - 0.5 * self.uL**2
        self.hR = self.htR - 0.5 * self.uR**2
        self.hAVG = alpha * self.hL + (1 - alpha) * self.hR + 0.5 * alpha * (1.0 - alpha) * self.deltaU**2

        # compute mean initial guess state
        p_mean = 0.5 * (self.pL + self.pR)
        rho_mean = 0.5 * (self.rhoL + self.rhoR)
        rhoeL = self.rhoL * self.eL
        rhoeR = self.rhoR * self.eR

        def _compute_chi_kappa_array(fluid, p, rho):
            """Evaluate Vinokur chi/kappa arrays."""
            p = np.asarray(p, dtype=float)
            rho = np.asarray(rho, dtype=float)
            chi, kappa = fluid.computeChiKappa_VinokurScheme_p_rho(p, rho)
            return np.asarray(chi, dtype=float), np.asarray(kappa, dtype=float)

        chiL, kappaL = _compute_chi_kappa_array(self.fluid, self.pL, self.rhoL)
        chiR, kappaR = _compute_chi_kappa_array(self.fluid, self.pR, self.rhoR)
        chiM, kappaM = _compute_chi_kappa_array(self.fluid, p_mean, rho_mean)
        chiHat = (chiL + chiR + 4.0 * chiM) / 6.0
        kappaHat = (kappaL + kappaR + 4.0 * kappaM) / 6.0
        delta_rhoe = rhoeR - rhoeL

        # projection procedure to compute the average state starting from the initial guess (hat values)
        error_term = self.deltaP - chiHat * self.deltaRho - kappaHat * delta_rhoe
        hM = 0.5 * (self.hL + self.hR)
        csquare_L = chiL + kappaL * self.hL
        csquare_R = chiR + kappaR * self.hR
        csquare_M = chiM + kappaM * hM
        sHat = (csquare_L + csquare_R + 4.0 * csquare_M) / 6.0
        D_term = (sHat * self.deltaRho)**2 + self.deltaP**2
        denom = D_term - self.deltaP * error_term

        self.chiAVG = np.where(
            self.deltaRho == 0,
            chiHat,
            (D_term * chiHat + sHat**2 * self.deltaRho * error_term) / denom,
        )
        self.kappaAVG = np.where(
            self.deltaP == 0,
            kappaHat,
            (D_term * kappaHat) / denom,
        )

        self.aAVG = np.sqrt(self.chiAVG + self.kappaAVG * self.hAVG)

    def computeFlux(self, entropyFixActive, fixCoefficient):
        """
        compute the global flux, average + dissipation. Returns shape (nFaces, 3),
        or (3,) for scalar input.
        """
        self.computeAveragedVariables()

        fluxL = self._assembleFlux(self.rhoL, self.uL, self.pL, self.eL)
        fluxR = self._assembleFlux(self.rhoR, self.uR, self.pR, self.eR)

        k1 = 0.5 * self.kappaAVG * self.uAVG**2 + self.kappaAVG
        k2 = 0.5 * self.uAVG**2 - self.chiAVG / self.kappaAVG

        n = self.nFaces
        matrixR = np.zeros((n, 3, 3))
        matrixR[:, 0, 0] = 1
        matrixR[:, 0, 1] = 1
        matrixR[:, 0, 2] = 1
        matrixR[:, 1, 0] = self.uAVG
        matrixR[:, 1, 1] = self.uAVG + self.aAVG
        matrixR[:, 1, 2] = self.uAVG - self.aAVG
        matrixR[:, 2, 0] = k2
        matrixR[:, 2, 1] = self.htAVG + self.aAVG * self.uAVG
        matrixR[:, 2, 2] = self.htAVG - self.aAVG * self.uAVG

        matrixRinv = np.zeros((n, 3, 3))
        matrixRinv[:, 0, 0] = 1 - k1 / self.aAVG**2
        matrixRinv[:, 0, 1] = self.kappaAVG * self.uAVG / self.aAVG**2
        matrixRinv[:, 0, 2] = -self.kappaAVG / self.aAVG**2
        matrixRinv[:, 1, 0] = 0.5 * (k1 / self.aAVG**2 - self.uAVG / self.aAVG)
        matrixRinv[:, 1, 1] = -0.5 * (self.kappaAVG * self.uAVG / self.aAVG**2 - 1 / self.aAVG)
        matrixRinv[:, 1, 2] = 0.5 * self.kappaAVG / self.aAVG**2
        matrixRinv[:, 2, 0] = 0.5 * (k1 / self.aAVG**2 + self.uAVG / self.aAVG)
        matrixRinv[:, 2, 1] = -0.5 * (self.kappaAVG * self.uAVG / self.aAVG**2 + 1 / self.aAVG)
        matrixRinv[:, 2, 2] = 0.5 * self.kappaAVG / self.aAVG**2

        eigsAVG = np.column_stack((self.uAVG, self.uAVG + self.aAVG, self.uAVG - self.aAVG))
        if entropyFixActive == False:
            absEig = np.abs(eigsAVG)
        else:
            absEig = applyEntropyFix(eigsAVG, self.aAVG, fixCoefficient)

        deltaCons = np.column_stack((self.u1R - self.u1L, self.u2R - self.u2L, self.u3R - self.u3L))
        projected = np.einsum('nij,nj->ni', matrixRinv, deltaCons)
        projected = projected * absEig
        deltaFlux = np.einsum('nij,nj->ni', matrixR, projected)

        fluxRoe = 0.5 * (fluxL + fluxR) - 0.5 * deltaFlux
        return fluxRoe

    

def applyEntropyFix(eigs, aAVG, kappa):
    """
    Apply Harten entropy fix to eigenvalues.

    eigs : ndarray of shape (3,) or (nFaces, 3)
        Raw Roe eigenvalues.
    aAVG : float or ndarray of shape (nFaces,)
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