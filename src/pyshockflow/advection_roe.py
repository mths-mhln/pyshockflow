import numpy as np
from numpy import sqrt
from pyshockflow import FluidIdeal
from pyshockflow.math_utils import *




def computeFluxRoeBaseMUSCL(rhoL, rhoR, uL, uR, pL, pR, fluid, entropyFixActive, fixCoefficient):
    """Compute Roe fluxes for all faces for the ideal-gas Roe scheme. 
    Formulation of the flux computation based on x-split Riemann Solver 
    in the book by Toro.

    Fluid states are MUSCL-reconstructed. In this scenario, 
    the left and right fluid state arrays are compltely different,
    and the internal energy values computed at the start of every iteration
    in the Driver.solve() method cannot be reused. """
    # compute left and right internal energy from the left and right
    # MUSCL-reconstructed pressure and density. 
    eL = fluid.computeInternalEnergy_p_rho(pL, rhoL)
    eR = fluid.computeInternalEnergy_p_rho(pR, rhoR)

    # compute the total enthalpy from its definition. 
    htL = 0.5 * uL**2 + eL + pL / rhoL
    htR = 0.5 * uR**2 + eR + pR / rhoR

    # precompute often reused terms for the Roe average state.
    sqrtRhoL = np.sqrt(rhoL)
    sqrtRhoR = np.sqrt(rhoR)
    denom = sqrtRhoL + sqrtRhoR

    # compute the Roe averaged variables for the 1D Euler equations
    rhoAVG = np.sqrt(rhoL * rhoR)
    uAVG = (sqrtRhoL * uL + sqrtRhoR * uR) / denom
    hAVG = (sqrtRhoL * htL + sqrtRhoR * htR) / denom
    aAVG = np.sqrt((fluid.gmma - 1.0) * (hAVG - 0.5 * uAVG**2))

    # compute eigenvalues of the averaged Jacobian.
    eigs = np.column_stack((uAVG - aAVG, uAVG, uAVG + aAVG))
    if entropyFixActive:
        absEig = applyEntropyFix(eigs, aAVG, fixCoefficient)
    else:
        absEig = np.abs(eigs)

    # Characteristic jumps due to initial conditions. Some delta's
    # are precomputed for efficiency.
    deltaP = pR - pL
    deltaU = uR - uL
    a2 = aAVG**2
    alpha0 = 0.5 / a2 * (deltaP - rhoAVG * aAVG * deltaU)
    alpha1 = (rhoR - rhoL) - deltaP / a2
    alpha2 = 0.5 / a2 * (deltaP + rhoAVG * aAVG * deltaU)

    # "Euler fluxes from MUSCL-reconstructed fluid States
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

    # compute the dissipation terms for the Roe fluxes. 
    # the dissipation terms are a combination of characteristic jump info, 
    # eigenvalue info, and eigenvector info, the latter of which is not 
    # explicitly computed here, but embedded in the calculation of the dissipation terms.
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



def computeFluxRoeBaseNoMUSCL(rhoL, rhoR, uL, uR, pL, pR, fluidState, fluid, entropyFixActive, fixCoefficient):
    """Compute Roe fluxes for all faces for the ideal-gas Roe scheme. 
    Formulation of the flux computation based on x-split Riemann Solver 
    in the book by Toro.

    Fluid states are not MUSCL-reconstructed. In this scenario, 
    the left and right fluid state arrays contain N-1 similar values, 
    with N being the total amount of elements in the array. The internal energy 
    values computed at the start of every iteration in the Driver.solve() 
    method _can_ be reused. """
    # unpack the fluid state dictionary for easier access to the variables
    # and reducing the amount of dictionary lookups.
    e = fluidState["staticInternalEnergy"]
    u = fluidState["Velocity"]
    p = fluidState["Pressure"]
    rho = fluidState["Density"]

    # Internal energy not computed since it has already been 
    # pre-computed in the Driver.solve() method.

    # compute the total enthalpy from its definition.
    ht = 0.5 * u**2 + e + p / rho

    # precompute often reused terms for the Roe average state.
    sqrtRhoL = np.sqrt(rhoL)
    sqrtRhoR = np.sqrt(rhoR)
    denom = sqrtRhoL + sqrtRhoR

    # compute the Roe averaged variables for the 1D Euler equations
    rhoAVG = np.sqrt(rhoL * rhoR)
    uAVG = (sqrtRhoL * uL + sqrtRhoR * uR) / denom
    hAVG = (sqrtRhoL * ht[:-1] + sqrtRhoR * ht[1:]) / denom
    aAVG = np.sqrt((fluid.gmma - 1.0) * (hAVG - 0.5 * uAVG**2))

    # compute eigenvalues of the averaged Jacobian.
    eigs = np.column_stack((uAVG - aAVG, uAVG, uAVG + aAVG))
    if entropyFixActive:
        absEig = applyEntropyFix(eigs, aAVG, fixCoefficient)
    else:
        absEig = np.abs(eigs)

    # Characteristic jumps due to initial conditions. Some delta's
    # are precomputed for efficiency.
    deltaP = pR - pL
    deltaU = uR - uL
    a2 = aAVG**2
    alpha0 = 0.5 / a2 * (deltaP - rhoAVG * aAVG * deltaU)
    alpha1 = (rhoR - rhoL) - deltaP / a2
    alpha2 = 0.5 / a2 * (deltaP + rhoAVG * aAVG * deltaU)

    # Euler fluxes from fluid States
    flux = np.column_stack((
        rho * u,
        rho * u**2 + p,
        u * (rho * (e + 0.5 * u**2) + p),
    ))

    # compute the dissipation terms for the Roe fluxes.
    # the dissipation terms are a combination of characteristic jump info, 
    # eigenvalue info, and eigenvector info, the latter of which is not 
    # explicitly computed here, but embedded in the calculation of the dissipation terms.
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

    return 0.5 * (flux[:-1] + flux[1:]) - 0.5 * diss



def computeFluxRoeArabiMUSCL(rhoL, rhoR, uL, uR, pL, pR, fluid, entropyFixActive, fixCoefficient):
    """Generalised Roe Scheme for real gases, taken from the article 
    'A simple extension of Roe scheme for real gases', Arabi et al. 
    Journal of Computational Physics 2017. Formulation based on 1D problem.
    
    Fluid states are MUSCL-reconstructed. In this scenario, 
    the left and right fluid state arrays are compltely different,
    and the internal energy values computed at the start of every iteration
    in the Driver.solve() method cannot be reused. """
    # compute left and right internal energy from the left and right
    # MUSCL-reconstructed pressure and density.
    eL = fluid.computeInternalEnergy_p_rho(pL, rhoL)
    eR = fluid.computeInternalEnergy_p_rho(pR, rhoR)

    # compute the total enthalpy from its definition.
    htL = 0.5 * uL**2 + eL + pL / rhoL
    htR = 0.5 * uR**2 + eR + pR / rhoR

    # precompute often reused terms for the Roe average state.
    n_faces = rhoL.size
    p_lr = np.concatenate((pL, pR))
    rho_lr = np.concatenate((rhoL, rhoR))
    a_lr = fluid.computeSoundSpeed_p_rho(p_lr, rho_lr)
    aL = a_lr[:n_faces]
    aR = a_lr[n_faces:]

    # precompute often reused terms for the Roe average state.
    sqrtRhoL = np.sqrt(rhoL)
    sqrtRhoR = np.sqrt(rhoR)
    denom = sqrtRhoL + sqrtRhoR

    # compute the Roe averaged variables for the 1D Euler equations
    rhoAVG = np.sqrt(rhoL * rhoR)
    uAVG = (sqrtRhoL * uL + sqrtRhoR * uR) / denom
    hAVG = (sqrtRhoL * htL + sqrtRhoR * htR) / denom
    aAVG = (sqrtRhoL * aL + sqrtRhoR * aR) / denom

    # compute the characteristic jumps due to initial conditions. Some delta's
    # are precomputed for efficiency.
    deltaP = pR - pL
    deltaU = uR - uL
    deltaRho = rhoR - rhoL
    a2 = aAVG**2
    alpha0 = 0.5 / a2 * (deltaP + rhoAVG * aAVG * deltaU)
    alpha1 = 0.5 / a2 * (deltaP - rhoAVG * aAVG * deltaU)
    alpha2 = deltaRho - deltaP / a2

    # compute the eigenvalues of the Roe-averaged Jacobian. 
    eigs = np.column_stack((uAVG + aAVG, uAVG - aAVG, uAVG))
    if entropyFixActive:
        absEig = applyEntropyFix(eigs, aAVG, fixCoefficient)
    else:
        absEig = np.abs(eigs)

    # Euler fluxes from the left and right MUSCL-reconstructed states.
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

    # compute the dissipation terms for the Roe fluxes.
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


def computeFluxRoeArabiNoMUSCL(rhoL, rhoR, uL, uR, pL, pR, fluidState, entropyFixActive, fixCoefficient):
    """Generalised Roe Scheme for real gases, taken from the article 
    'A simple extension of Roe scheme for real gases', Arabi et al. 
    Journal of Computational Physics 2017. Formulation based on 1D problem.
        
    Fluid states are not MUSCL-reconstructed. In this scenario, 
    the left and right fluid state arrays contain N-1 similar values, 
    with N being the total amount of elements in the array. The internal energy 
    values computed at the start of every iteration in the Driver.solve() 
    method _can_ be reused."""
    # unpack the fluid state dictionary for easier access to the variables
    # and reducing the amount of dictionary lookups.
    e = fluidState["staticInternalEnergy"]
    u = fluidState["Velocity"]
    p = fluidState["Pressure"]
    rho = fluidState["Density"]

    # Internal energy not computed since it has already been 
    # pre-computed in the Driver.solve() method.

    # compute the total enthalpy from its definition.
    ht = 0.5 * u**2 + e + p / rho

    # Sound Speed not computed since it has already been 
    # pre-computed in the Driver.solve() method.
    aL = fluidState["soundSpeed"][:-1]
    aR = fluidState["soundSpeed"][1:]

    # precompute often reused terms for the Roe average state.
    sqrtRhoL = np.sqrt(rhoL)
    sqrtRhoR = np.sqrt(rhoR)
    denom = sqrtRhoL + sqrtRhoR

    # compute the Roe averaged variables for the 1D Euler equations
    rhoAVG = np.sqrt(rhoL * rhoR)
    uAVG = (sqrtRhoL * uL + sqrtRhoR * uR) / denom
    hAVG = (sqrtRhoL * ht[:-1] + sqrtRhoR * ht[1:]) / denom
    aAVG = (sqrtRhoL * aL + sqrtRhoR * aR) / denom

    # compute the characteristic jumps due to initial conditions. Some delta's
    # are precomputed for efficiency.
    deltaP = pR - pL
    deltaU = uR - uL
    deltaRho = rhoR - rhoL
    a2 = aAVG**2
    alpha0 = 0.5 / a2 * (deltaP + rhoAVG * aAVG * deltaU)
    alpha1 = 0.5 / a2 * (deltaP - rhoAVG * aAVG * deltaU)
    alpha2 = deltaRho - deltaP / a2

    # compute the eigenvalues of the Roe-averaged Jacobian.
    eigs = np.column_stack((uAVG + aAVG, uAVG - aAVG, uAVG))
    if entropyFixActive:
        absEig = applyEntropyFix(eigs, aAVG, fixCoefficient)
    else:
        absEig = np.abs(eigs)

    # euler fluxes from the left and right fluid states.
    flux = np.column_stack((
        rho * u,
        rho * u**2 + p,
        u * (rho * (e + 0.5 * u**2) + p),
    ))

    deltaF = np.zeros_like(flux[:-1])
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
        (rhoR * uR * ht[1:])
        - (rhoL * uL * ht[:-1])
        - (hAVG + uAVG * aAVG) * (uAVG + aAVG) * (0.5 / a2 * (deltaP + rhoAVG * aAVG * deltaU))
        - (hAVG - uAVG * aAVG) * (uAVG - aAVG) * (0.5 / a2 * (deltaP - rhoAVG * aAVG * deltaU))
    )
    X = np.where(uAVG >= 0.0, X, -X)
    deltaF[:, 2] = (
        (hAVG + uAVG * aAVG) * absEig[:, 0] * alpha0
        + (hAVG - uAVG * aAVG) * absEig[:, 1] * alpha1
        + X
    )

    return 0.5 * (flux[:-1] + flux[1:]) - 0.5 * deltaF
    



def computeFluxRoeVinokurMUSCL(rhoL, rhoR, uL, uR, pL, pR, fluid, entropyFixActive, fixCoefficient):
    """Generalised Roe Scheme for real gases, 
    where the Roe avg state is taken from the article 
    'Generalized flux-vector splitting and Roe average for an equilibrium real gas', 
    Vinokur and Montagnè Journal of Computational Physics 1990.
    Formulation based on 1D problem.

    Fluid states are MUSCL-reconstructed. In this scenario, 
    the left and right fluid state arrays are compltely different,
    and the internal energy values computed at the start of every iteration
    in the Driver.solve() method cannot be reused. """
    # compute left and right internal energy from the left and right
    # MUSCL-reconstructed pressure and density.
    eL = fluid.computeInternalEnergy_p_rho(pL, rhoL)
    eR = fluid.computeInternalEnergy_p_rho(pR, rhoR)

    # compute the total enthalpy from its definition.
    htL = 0.5 * uL**2 + eL + pL / rhoL
    htR = 0.5 * uR**2 + eR + pR / rhoR

    # precompute often reused terms for the Roe average state.
    deltaP = pR - pL
    deltaU = uR - uL
    deltaRho = rhoR - rhoL
    sqrtRhoL = np.sqrt(rhoL)
    sqrtRhoR = np.sqrt(rhoR)
    alpha = sqrtRhoL / (sqrtRhoL + sqrtRhoR)

    # compute the Roe averaged variables for the 1D Euler equations
    uAVG = alpha * uL + (1.0 - alpha) * uR
    htAVG = alpha * htL + (1.0 - alpha) * htR
    hL = htL - 0.5 * uL**2
    hR = htR - 0.5 * uR**2
    hAVG = alpha * hL + (1.0 - alpha) * hR + 0.5 * alpha * (1.0 - alpha) * deltaU**2

    # compute mean initial guess state
    p_mean = 0.5 * (pL + pR)
    rho_mean = 0.5 * (rhoL + rhoR)
    rhoeL = rhoL * eL
    rhoeR = rhoR * eR

    def _compute_chi_kappa_array(fluid, p, rho):
        """Evaluate Vinokur chi/kappa arrays."""
        p = np.asarray(p, dtype=float)
        rho = np.asarray(rho, dtype=float)
        chi, kappa = fluid.computeChiKappa_VinokurScheme_p_rho(p, rho)
        return np.asarray(chi, dtype=float), np.asarray(kappa, dtype=float)
    chiL, kappaL = _compute_chi_kappa_array(fluid, pL, rhoL)
    chiR, kappaR = _compute_chi_kappa_array(fluid, pR, rhoR)
    chiM, kappaM = _compute_chi_kappa_array(fluid, p_mean, rho_mean)

    chiHat = (chiL + chiR + 4.0 * chiM) / 6.0
    kappaHat = (kappaL + kappaR + 4.0 * kappaM) / 6.0
    delta_rhoe = rhoeR - rhoeL

    # projection procedure to compute the average state starting fro the initial guess (hat values)
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

    # Euler fluxes from the MUSCL-reconstructed fluid states
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

    # compute the Eigenvectors matrices
    k1 = 0.5 * kappaAVG * uAVG**2 + kappaAVG
    k2 = 0.5 * uAVG**2 - chiAVG / kappaAVG

    # right eigenvectors matrix
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

    # left eigenvectors matrix
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

    # eigenvalues, to fix
    eigs = np.column_stack((uAVG, uAVG + aAVG, uAVG - aAVG))
    if entropyFixActive:
        absEig = applyEntropyFix(eigs, aAVG, fixCoefficient)
    else:
        absEig = np.abs(eigs)

    # compute the Flux
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




def computeFluxRoeVinokurNoMUSCL(rhoL, rhoR, uL, uR, pL, pR, fluidState, fluid, entropyFixActive, fixCoefficient):
    """Generalised Roe Scheme for real gases, 
    where the Roe avg state is taken from the article 
    'Generalized flux-vector splitting and Roe average for an equilibrium real gas', 
    Vinokur and Montagnè Journal of Computational Physics 1990.
    Formulation based on 1D problem.
    
    Fluid states are not MUSCL-reconstructed. In this scenario, 
    the left and right fluid state arrays contain N-1 similar values, 
    with N being the total amount of elements in the array. The internal energy 
    values computed at the start of every iteration in the Driver.solve() 
    method _can_ be reused."""
    # unpack the fluid state dictionary for easier access to the variables
    # and reducing the amount of dictionary lookups.
    e = fluidState["staticInternalEnergy"]
    u = fluidState["Velocity"]
    p = fluidState["Pressure"]
    rho = fluidState["Density"]

    # Internal energy not computed since it has already been
    # pre-computed in the Driver.solve() method.

    # compute the total enthalpy from its definition.
    ht = 0.5 * u**2 + e + p / rho

    # precompute often reused terms for the Roe average state.
    deltaP = pR - pL
    deltaU = uR - uL
    deltaRho = rhoR - rhoL
    sqrtRhoL = np.sqrt(rhoL)
    sqrtRhoR = np.sqrt(rhoR)
    alpha = sqrtRhoL / (sqrtRhoL + sqrtRhoR)

    # compute the Roe averaged variables for the 1D Euler equations
    uAVG = alpha * uL + (1.0 - alpha) * uR
    htAVG = alpha * ht[:-1] + (1.0 - alpha) * ht[1:]
    hL = ht[:-1] - 0.5 * uL**2
    hR = ht[1:] - 0.5 * uR**2
    hAVG = alpha * hL + (1.0 - alpha) * hR + 0.5 * alpha * (1.0 - alpha) * deltaU**2

    # compute mean initial guess state
    p_mean = 0.5 * (pL + pR)
    rho_mean = 0.5 * (rhoL + rhoR)
    rhoeL = rhoL * e[:-1]
    rhoeR = rhoR * e[1:]

    def _compute_chi_kappa_array(fluid, p, rho):
        """Evaluate Vinokur chi/kappa arrays."""
        p = np.asarray(p, dtype=float)
        rho = np.asarray(rho, dtype=float)
        chi, kappa = fluid.computeChiKappa_VinokurScheme_p_rho(p, rho)
        return np.asarray(chi, dtype=float), np.asarray(kappa, dtype=float)

    chi, kappa = _compute_chi_kappa_array(fluid, p, rho)
    chiM, kappaM = _compute_chi_kappa_array(fluid, p_mean, rho_mean)

    chiHat = (chi[:-1] + chi[1:] + 4.0 * chiM) / 6.0
    kappaHat = (kappa[:-1] + kappa[1:] + 4.0 * kappaM) / 6.0
    delta_rhoe = rhoeR - rhoeL

    # projection procedure to compute the average state starting fro the 
    # initial guess (hat values)
    error_term = deltaP - chiHat * deltaRho - kappaHat * delta_rhoe
    hM = 0.5 * (hL + hR)
    csquare_L = chi[:-1] + kappa[:-1] * hL
    csquare_R = chi[1:] + kappa[1:] * hR
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

    # Euler fluxes from the non-MUSCL-reconstructed fluid states
    flux = np.column_stack((
            rho * u,
            rho * u**2 + p,
            u * (rho * (e + 0.5 * u**2) + p),
        ))

    # compute the Eigenvectors matrices
    k1 = 0.5 * kappaAVG * uAVG**2 + kappaAVG
    k2 = 0.5 * uAVG**2 - chiAVG / kappaAVG

    # right eigenvectors matrix
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

    # left eigenvectors matrix
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


    # eigenvalues, to fix
    eigs = np.column_stack((uAVG, uAVG + aAVG, uAVG - aAVG))
    if entropyFixActive:
        absEig = applyEntropyFix(eigs, aAVG, fixCoefficient)
    else:
        absEig = np.abs(eigs)

    # compute the flux
    u1L = rhoL
    u2L = rhoL * uL
    u3L = rhoL * (0.5 * uL**2 + e[:-1])
    u1R = rhoR
    u2R = rhoR * uR
    u3R = rhoR * (0.5 * uR**2 + e[1:])
    deltaCons = np.column_stack((u1R - u1L, u2R - u2L, u3R - u3L))

    projected = np.einsum("nij,nj->ni", matrixRinv, deltaCons)
    projected *= absEig
    deltaFlux = np.einsum("nij,nj->ni", matrixR, projected)

    return 0.5 * (flux[:-1] + flux[1:]) - 0.5 * deltaFlux



        

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