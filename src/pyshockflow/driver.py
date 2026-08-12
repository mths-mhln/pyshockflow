import os
import re
import pickle
import sys
import copy
import shutil

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pyshockflow import Config
from pyshockflow import RiemannProblem
from pyshockflow import AdvectionRoeBase, AdvectionRoeArabi, AdvectionRoeVinokur
from pyshockflow import FluidIdeal, FluidReal

from pyshockflow.math_utils import (
    getConservativesFromFluidState,
    getFluidStateFromConservatives,
    computeAdvectionFluxFromConservatives
)


class Driver:
    def __init__(self, configFilePath=None, restartFilePath=None):
        """
        Initialize the Driver object for a new simulation or to continue a previous
        simulation from a restart file.

        Arguments
        ---------
        configFilePath : str, optional
            The path to the configuration file, which contains the simulation parameters.
            If not provided, a restart file must be specified.
        restartFilePath : str, optional
            The path to the restart file, which contains the simulation data from a
            previous step. If not provided, a configuration file must be specified.

        Returns
        -------
        None, stores all the relevant information as attributes of the Driver class.
        """
        if configFilePath is None and restartFilePath is None:
            raise ValueError(
                "Either a configuration file path or a restart file path must be provided."
            )
        if restartFilePath is not None:
            # Prepare Driver object to continue simulation from a previous step.
            # If user specifies a config file as well, the old simulation configuration
            # options stored in the restart file will be overwritten by the new ones.
            self.prepareRestart(configFilePath, restartFilePath)
        else:
            # Prepare Driver object to start a new simulation from scratch based
            # on the configuration options provided by the user.
            self.prepareCleanStart(configFilePath)


    # =========================================================================
    #  Startup helpers
    # =========================================================================

    def prepareCleanStart(self, configFilePath):
        """
        Prepare a fresh simulation from scratch: read config, build mesh, instantiate
        the fluid model, initialize the fluid state, and set boundary conditions.

        Arguments
        ---------
        configFilePath : str
            Path to the .ini configuration file.

        Returns
        -------
        None, but populates all attributes needed by solve().
        """
        config = Config(configFilePath)
        self._printWelcomeBanner(config)

        deviceGeometryData = self.extractDeviceGeometricalFeatures(config)
        meshData           = self.generateMesh(config, deviceGeometryData)
        fluidModel         = self.instantiateFluidModel(config)
        fluidState         = self.initializeFluidStateArrays(config, deviceGeometryData, meshData, fluidModel)
        fluidState         = self.setBoundaryConditions(config, meshData, fluidModel, fluidState)

        # Conservative variables are derived from the fluidState variables; initialise them
        # so that updateSolution() can operate on them from the very first step.
        conservativeState  = self._conservativesFromFluidState(fluidState, fluidModel)

        # Output paths — created fresh (never from a previous run).
        resultsPath = self._prepareOutputPaths(config, meshData, restartFilePath=None)

        # Pack everything the solver needs into attributes.
        self.config            = config
        self.meshData          = meshData
        self.deviceGeometryData = deviceGeometryData
        self.fluidModel        = fluidModel
        self.fluidState        = fluidState
        self.conservativeState = conservativeState
        self.resultsPath       = resultsPath
        self.time              = 0.0
        self.iterationIndex    = 0

        for attr, value in vars(self).items():
            print(f"{attr}: {value}")


    def prepareRestart(self, configFilePath, restartFilePath):
        """
        Prepare a simulation that continues from a previously saved restart file.
        The geometry, mesh, fluid model and output paths are rebuilt from config;
        the fluid state (and time counter) are taken from the restart file.

        Arguments
        ---------
        configFilePath : str or None
            Path to an (optionally) overriding .ini configuration file.  When
            None the configuration stored inside the restart file is used.
        restartFilePath : str
            Path to the pickle restart file produced by saveResults().

        Returns
        -------
        None, but populates all attributes needed by solve().
        """
        timeElapsed, fluidStateRestart, configRestart, iterationIndex = \
            self.extractRestartData(restartFilePath)

        print(
            f"Restarting simulation from file {restartFilePath} "
            f"at iteration {iterationIndex} and time elapsed {timeElapsed:.6e} s"
        )

        # Allow the user to override the stored config by supplying a new file.
        config = Config(configFilePath) if configFilePath is not None else configRestart

        self._printWelcomeBanner(config)

        deviceGeometryData = self.extractDeviceGeometricalFeatures(config)
        meshData           = self.generateMesh(config, deviceGeometryData)
        fluidModel         = self.instantiateFluidModel(config)

        # Fluid state comes from the restart file, not from initializeFluidStateArrays.
        fluidState         = fluidStateRestart
        fluidState         = self.setBoundaryConditions(config, meshData, fluidModel, fluidState)

        # Conservative variables are derived from the fluidState variables; initialise them
        # according to the restart fluid state so that updateSolution() can operate on them.
        conservativeState  = self._conservativesFromFluidState(fluidState, fluidModel)

        # Append new iterations to the same directory as the restart file.
        resultsPath = Path(restartFilePath).parent

        # Pack everything the solver needs into attributes.
        self.config             = config
        self.meshData           = meshData
        self.deviceGeometryData = deviceGeometryData
        self.fluidModel         = fluidModel
        self.fluidState         = fluidState
        self.conservativeState  = conservativeState
        self.resultsPath        = resultsPath
        self.time               = timeElapsed
        self.iterationIndex     = iterationIndex


    @staticmethod
    def _printWelcomeBanner(config):
        """Print the welcome banner and basic simulation information to stdout."""
        print("\n" + "=" * 80)
        print(" " * 25 + " WELCOME TO PYSHOCKTUBE ")
        print(" " * 18 + "Fluid Dynamics Simulation for Shock Tubes")
        print("=" * 80)
        print()
        print("=" * 80)
        print(" " * 32 + "SIMULATION DATA")
        print("Fluid name:                                  %s" % config.fluidName())
        print("Fluid treatment:                             %s" % config.fluidModel())
        if config.fluidModel().lower() == "ideal":
            print("Fluid cp/cv ratio [-]:                       %.6e" % config.fluidGamma())
            print("Fluid gas constant [J/kgK]:                  %.6e" % config.gasRConstant())
        print("Boundary Conditions Left:                    %s" % config.boundaryConditions()[0])
        print("Boundary Conditions Right:                   %s" % config.boundaryConditions()[1])
        print("=" * 80)


    # =========================================================================
    #  Geometry and mesh
    # =========================================================================

    def extractDeviceGeometricalFeatures(self, config):
        """
        Extract geometrical features of the expansion device under consideration from
        the user-specified CSV file. Depending on the expansion device under consideration
        the data extraction method differs. Please refer to the PyShockFlow input
        documentation for the required format of the expansion device geometry file,
        for each type of geometry.

        Arguments
        ---------
        config : Config
            The configuration object containing the simulation parameters.

        Returns
        -------
        Nozzle geometry:
            geometryData : dict
                A dictionary containing the extracted geometrical features.
                - deviceX : np.1darray
                    The physical distance along the nozzle.
                - deviceArea : np.1darray
                    The area of the nozzle cross-section at each point.

        Shock tube geometry:
            geometryData : dict
                A dictionary containing the extracted geometrical features.
                - deviceX : np.1darray
                    The physical length of the shock tube.
                - deviceArea : np.1darray
                    The area of the shock tube cross-section, assumed to be uniform.
                - shockTubeInterfaceLoc : float
                    The location of the interface between the high-pressure and
                    low-pressure regions.
        """
        # Extract the path to the CSV file containing the geometrical features from config.
        deviceGeometryFilePath = config.deviceGeometryFilePath()

        # Instantiate geometryData dictionary.
        deviceGeometryData = {}

        # Extract nozzle ordinates (physical distance along the nozzle) and coordinate
        # (area of the nozzle cross-section) and store those in the geometryData dictionary.
        nozzleData = np.loadtxt(deviceGeometryFilePath, skiprows=1, delimiter=",", dtype=float)
        deviceGeometryData["deviceX"]    = nozzleData[:, 0]
        deviceGeometryData["deviceArea"] = nozzleData[:, 1]

        # According to the shock tube input data format requirements, the second data row
        # (disregarding the header) contains the interface location.
        if config.expansionDeviceType() == "shocktube":
            deviceGeometryData["shockTubeInterfaceLoc"] = nozzleData[1, 0]

        # Generate some QoL information.
        deviceGeometryData["deviceLength"] = deviceGeometryData["deviceX"][-1] - deviceGeometryData["deviceX"][0]

        return deviceGeometryData


    def generateMesh(self, config, deviceGeometryData):
        """
        Build the 1D mesh node positions along the shock tube. Generation is based on the
        expansion device start and end coordinates (taken from the geometry data) and the
        mesh specifications provided by the user in the configuration file. The mesh consists
        of physical domain nodes and halo nodes, at which boundary conditions will be imposed.

        If mesh refinement is enabled, the refinement zone uses a fixed number of uniformly-
        spaced nodes (numMeshNodesRef). Outside the refinement zone, spacing transitions
        linearly from dx_refined (at the refinement boundary) to dx_uniform (the spacing that
        would result from placing `numMeshNodes` points uniformly over the full domain),
        ensuring a smooth transition without an abrupt jump.

        Arguments
        ---------
        config : Config
            The configuration object containing the mesh specifications.
        deviceGeometryData : dict
            A dictionary containing the geometrical features of the expansion device.

        Returns
        -------
        meshData : dict
            - xMeshNodes          : np.1darray — node positions including halo nodes
            - numMeshNodes        : int         — total number of nodes (physical + 2 halos)
            - meshNodeSpacing     : np.1darray  — cell width at each node
            - deviceAreaAtMeshNodes : np.1darray — cross-sectional area interpolated at nodes
            - dAreaDx             : np.1darray  — area gradient along x, used for source terms
        """
        def _build_outside_section(start, end, n_points, dx_near, dx_far, direction):
            """
            Build a 1D mesh section with linearly varying spacing, transitioning smoothly
            from dx_near (at the refinement boundary) to dx_far (at the domain edge).

            Arguments
            ---------
            start, end : float
                Endpoints of this section.
            n_points : int
                Number of nodes including both endpoints.
            dx_near : float
                Spacing at the refinement boundary.
            dx_far : float
                Spacing at the domain edge.
            direction : str
                'upstream' (refinement on the right) or 'downstream' (refinement on the left).

            Returns
            -------
            coords : numpy array
                Node positions for this section.
            """
            if n_points <= 1:
                return np.array([start])

            # Spacing array runs from dx_far (domain edge) to dx_near (refinement side),
            # so the finest spacing is always adjacent to the refinement zone.
            spacings = np.linspace(dx_far, dx_near, n_points - 1)

            if direction == "upstream":
                # Refinement is on the right: build right-to-left from the refinement
                # boundary, then reverse so coords run left-to-right.
                coords = [end]
                for dx in reversed(spacings):
                    coords.append(coords[-1] - dx)
                coords = np.array(list(reversed(coords)))
            else:
                # Refinement is on the left: build left-to-right from the refinement boundary.
                coords = [start]
                for dx in spacings:
                    coords.append(coords[-1] + dx)
                coords = np.array(coords)

            # Rescale to enforce exact endpoints, correcting for floating-point drift
            # without altering the relative node distribution.
            coords = start + (coords - coords[0]) / (coords[-1] - coords[0]) * (end - start)

            return coords

        # Instantiate meshData dictionary.
        meshData = {}

        # Extract length of the expansion device computational domain and number
        # of mesh nodes the user wants to place in it.
        length       = deviceGeometryData["deviceLength"]
        numMeshNodes = config.numberOfMeshNodes()

        # Check if mesh refinement is enabled in the configuration file.
        isMeshRefined = config.meshRefinementBool()

        if not isMeshRefined:
            xMeshNodes = np.linspace(deviceGeometryData["deviceX"][0], deviceGeometryData["deviceX"][-1], numMeshNodes)
        else:
            refinementCoords = config.refinementBoundaries()
            print("Mesh is refined between the two boundaries [m]: ", refinementCoords)

            numMeshNodesRef = config.numberOfRefMeshNodes()
            x0_ref, x1_ref = refinementCoords

            # dx_refined: uniform spacing inside the refinement zone, used as the fine
            # anchor for the linear interpolation in the outside sections.
            dx_refined = (x1_ref - x0_ref) / numMeshNodesRef

            # dx_uniform: spacing of a hypothetical uniform mesh with `numMeshNodes` points
            # over the full domain, used as the coarse anchor for the linear interpolation.
            dx_uniform = length / (numMeshNodes - 1)

            # Build the refinement zone; endpoints are shared with the outside sections.
            xRefinement = np.linspace(x0_ref, x1_ref, numMeshNodesRef + 1)

            lengthUpstream   = x0_ref
            lengthDownstream = length - x1_ref
            lengthTotal      = lengthUpstream + lengthDownstream

            has_upstream   = lengthUpstream   > 0
            has_downstream = lengthDownstream > 0

            if has_upstream and has_downstream:
                # Split `numMeshNodes` proportionally by section length.
                n_upstream   = max(2, round(numMeshNodes * lengthUpstream   / lengthTotal))
                n_downstream = max(2, round(numMeshNodes * lengthDownstream / lengthTotal))

                xUpstream   = _build_outside_section(
                    0, x0_ref, n_upstream, dx_refined, dx_uniform, "upstream"
                )
                xDownstream = _build_outside_section(
                    x1_ref, length, n_downstream, dx_refined, dx_uniform, "downstream"
                )

                # Drop the last node of xUpstream and xRefinement to avoid duplicating
                # the shared boundary nodes at x0_ref and x1_ref.
                xMeshNodes = np.concatenate((xUpstream[:-1], xRefinement[:-1], xDownstream))

            elif has_upstream:
                # Refinement ends at the right edge of the domain.
                xUpstream = _build_outside_section(
                    0, x0_ref, max(2, numMeshNodes), dx_refined, dx_uniform, "upstream"
                )
                xMeshNodes = np.concatenate((xUpstream[:-1], xRefinement))

            elif has_downstream:
                # Refinement starts at the left edge of the domain.
                xDownstream = _build_outside_section(
                    x1_ref, length, max(2, numMeshNodes), dx_refined, dx_uniform, "downstream"
                )
                xMeshNodes = np.concatenate((xRefinement[:-1], xDownstream))

            else:
                raise ValueError(
                    "The refinement zone covers the entire domain. "
                    "Nothing is left to mesh outside it."
                )

        # Physical domain mesh is built; add halo nodes.  A halo node is placed one
        # cell-width beyond each end of the physical domain.
        xHaloLeft  = xMeshNodes[0]  - (xMeshNodes[1]  - xMeshNodes[0])
        xHaloRight = xMeshNodes[-1] + (xMeshNodes[-1] - xMeshNodes[-2])
        xMeshNodes = np.concatenate(([xHaloLeft], xMeshNodes, [xHaloRight]))

        # Save physical positions of mesh nodes in the meshData dictionary.
        meshData["xMeshNodes"] = xMeshNodes

        # Total number of mesh nodes (physical + 2 halo nodes).
        meshData["numMeshNodes"] = len(xMeshNodes)

        # meshNodeSpacing: the physical width of each mesh cell.  np.gradient gives
        # second-order central differences in the interior and one-sided differences
        # at the boundaries, which is exactly what we want.
        def computeGridSpacing(xNodes):
            """
            Compute spacing between the physical geometry nodes, which is needed for the solver and for the mesh stretching if activated.

            Arguments
            ---------
            xNodes : numpy array
                The coordinates of the nodes along the 1D (x) axis, excluding halo nodes.

            Returns
            -------
            dx : numpy array
                The spacing between the nodes along the 1D (x) axis, excluding halo nodes.
            """
            dx = np.zeros_like(xNodes)
            dx[0] = xNodes[1]-xNodes[0]
            for i in range(1,len(dx)-1):
                dx[i] = (xNodes[i+1]-xNodes[i])/2 + (xNodes[i]-xNodes[i-1])/2
            dx[-1] = xNodes[-1]-xNodes[-2]
            return dx
        # meshData["meshNodeSpacing"] = np.gradient(xMeshNodes)
        meshData["meshNodeSpacing"] = computeGridSpacing(xMeshNodes)

        # Interpolate the device cross-sectional area variation at the mesh node
        # locations.  Keeping the area constant outside the physical domain prevents
        # spurious area gradients from being extrapolated into the halo nodes.
        meshData["deviceAreaAtMeshNodes"] = np.interp(
            xMeshNodes,
            deviceGeometryData["deviceX"],
            deviceGeometryData["deviceArea"],
            left=deviceGeometryData["deviceArea"][0],
            right=deviceGeometryData["deviceArea"][-1],
        )

        # Pre-compute the area gradient along x; it appears in the quasi-1D source
        # terms and is cheaper to compute once here than every residual evaluation.
        meshData["dAreaDx"] = np.gradient(
            meshData["deviceAreaAtMeshNodes"], xMeshNodes
        )

        return meshData


    # =========================================================================
    #  Fluid model
    # =========================================================================

    def instantiateFluidModel(self, config):
        """
        Instantiate the fluid model based on the user-specified configuration options.
        The fluid model is used to compute thermodynamic properties of the working fluid,
        such as density, pressure, temperature and energy.

        Arguments
        ---------
        config : Config
            The configuration object containing the simulation parameters.

        Returns
        -------
        fluid : FluidIdeal or FluidReal
            An instance of the fluid model class, capable of computing thermodynamic
            properties.
        """
        fluidModel = config.fluidModel()

        if fluidModel.lower() == "ideal":
            gmma = config.fluidGamma()
            Rgas = config.gasRConstant()
            return FluidIdeal(gmma, Rgas)

        elif fluidModel.lower() == "real":
            fluidName    = config.fluidName()
            fluidLibrary = config.fluidLibrary()
            return FluidReal(fluidName, fluidLibrary, config.propertyExtractionMethod(), False)
        


    # =========================================================================
    #  Initial conditions
    # =========================================================================

    def initializeFluidStateArrays(self, config, deviceGeometryData, meshData, fluidModel):
        """
        Initialize fluid state arrays at the mesh nodes. The fluid state arrays comprise
        four variables of interest: density, velocity, pressure and energy.

        For nozzle geometries, the fluid thermodynamic state is initialized in different
        ways depending on the boundary conditions specified by the user. If inlet and
        outlet, or inlet and transparent boundary conditions are specified, the fluid
        thermodynamic state is initialized linearly between the boundary conditions;
        otherwise, the state assumes the boundary conditions throughout the geometry.

        For shocktube geometries, the fluid thermodynamic state is initialized based on
        user-specified initial conditions that specify the thermodynamic state of the
        working fluid on each side of the interface. Conditions are assumed to be
        uniform on each side of the interface.

        Arguments
        ---------
        config : Config
            The configuration object containing the simulation parameters.
        meshData : dict
            Mesh data dictionary produced by generateMesh().
        fluidModel : FluidIdeal or FluidReal
            An instance of the fluid model class.

        Returns
        -------
        fluidState : dict
            Dictionary with keys 'Density', 'Velocity', 'Pressure', 'Energy',
            each mapping to a np.1darray of length meshData['numMeshNodes'].
        """
        # Instantiate fluid state dict and arrays.
        fluidState = {
            key: np.zeros(meshData["numMeshNodes"])
            for key in ("Density", "Velocity", "Pressure", "Energy")
        }

        # ------------------------------------------------------------------
        # Helper: shocktube initialization
        # ------------------------------------------------------------------
        def _imposeInitialConditionsShocktube(config, deviceGeometryData, meshData, fluidModel, fluidState):
            """
            Initialize fluid state variables on either side of the interface for shocktube
            experiments.  Thermodynamic state is specified via (p, rho) if density is
            provided in the config, otherwise via (p, T).  The interface is imposed
            through a vectorised np.where on the mesh node positions.

            Arguments
            ---------
            config : Config
            fluidModel : FluidIdeal or FluidReal
            fluidState : dict (modified in-place and returned)

            Returns
            -------
            fluidState : dict
            """
            pL, pR = config.initialPressureLeft(), config.initialPressureRight()
            vL, vR = config.initialVelocityLeft(), config.initialVelocityRight()

            # Use density if provided, otherwise temperature; compute the other from
            # the specified quantity together with the pressure.
            try:
                rhoL, rhoR = config.initialDensityLeft(), config.initialDensityRight()
                TL = fluidModel.computeTemperature_p_rho(pL, rhoL)
                TR = fluidModel.computeTemperature_p_rho(pR, rhoR)
            except Exception as e:
                TL, TR = config.initialTemperatureLeft(), config.initialTemperatureRight()
                rhoL   = fluidModel.computeDensity_p_T(pL, TL)
                rhoR   = fluidModel.computeDensity_p_T(pR, TR)

            # Compute static internal energy from the thermodynamic state.
            eL = fluidModel.computeInternalEnergy_p_rho(pL, rhoL)
            eR = fluidModel.computeInternalEnergy_p_rho(pR, rhoR)

            # Group in dictionary to simplify the loop below.
            initialConditions = {
                "Density":  (rhoL, rhoR),
                "Velocity": (vL,   vR),
                "Pressure": (pL,   pR),
                "Energy":   (eL,   eR),
            }

            xInterface = deviceGeometryData["shockTubeInterfaceLoc"]
            for key, (valueL, valueR) in initialConditions.items():
                fluidState[key] = np.where(
                    meshData["xMeshNodes"] <= xInterface, valueL, valueR
                )
            print(f"Initial L/R density values [kg/m3]:     ({rhoL:.6e}, {rhoR:.6e})")
            print(f"Initial L/R velocity values [m/s]:      ({vL:.6e}, {vR:.6e})")
            print(f"Initial L/R pressure values [Pa]:       ({pL:.6e}, {pR:.6e})")
            print(f"Initial L/R temperature values [K]:     ({TL:.6e}, {TR:.6e})")
            print(f"Initial L/R energy values [J/kg]:       ({eL:.6e}, {eR:.6e})")

            return fluidState

        # ------------------------------------------------------------------
        # Helper: linear nozzle initialization
        # ------------------------------------------------------------------
        def _imposeInitialConditionsNozzleLinear(config, meshData, fluidModel, fluidState):
            """
            Method reserved for nozzle flow experiments.  Following the advice of
            Cioffi et al. (2025) "A Hyperbolic One-Dimensional Model for Two-Phase
            Flows in Converging-Diverging Nozzles", the flow field is initialized in
            a linear fashion from inlet to outlet.

            Arguments
            ---------
            config : Config
            meshData : dict
            fluidModel : FluidIdeal or FluidReal
            fluidState : dict (modified in-place and returned)

            Returns
            -------
            fluidState : dict
            """
            isTotalInlet = config.inletConditionsType().lower() == "total"
            isStaticInlet = config.inletConditionsType().lower() == "static"
            inletIdx = None  # index (0 or -1) of whichever side is the inlet

            # Determine which type of total inlet variables were specified so that
            # the correct initialization can be performed further below.
            inletConditionsValues = config.inletConditionsValues()
            if isTotalInlet:
                if 0.0 <= inletConditionsValues[1] <= 1.0:
                    inletConditionsVars = "ptQt"
                else:
                    inletConditionsVars = "ptTt"
            if isStaticInlet:
                if 0.0 <= inletConditionsValues[1] <= 1.0:
                    inletConditionsVars = "pQ"
                else:
                    inletConditionsVars = "pT"

            # Impose boundary conditions at each edge of the domain, tracking which
            # edge is the inlet.
            for iHalo, iInternal in [(0, 1), (-1, -2)]:
                btype = config.boundaryConditions()[0 if iHalo == 0 else 1].lower()

                if btype == "inlet":
                    if isTotalInlet:
                        # setInletBoundaryConditions needs a static pressure guess for
                        # its iteration to converge; seed the neighbouring node with the
                        # (total) inlet pressure as a starting value.
                        fluidState["Pressure"][iInternal] = inletConditionsValues[0]

                    if isStaticInlet:
                        # setInletBoundaryConditions needs a velocity guess to extrapolate
                        # from the internal domain. 
                        # Static inlet conditions: velocity initialized linearly from 10 to 200 m/s.
                        fluidState["Velocity"] = np.interp(
                            meshData["xMeshNodes"],
                            [meshData["xMeshNodes"][0], meshData["xMeshNodes"][-1]],
                            [10, 200],
                        )
                    
                    fluidState = _applyInletBC(
                        iHalo, iInternal, fluidModel, fluidState,
                        isTotalInlet, inletConditionsVars if isTotalInlet else None,
                        inletConditionsValues,
                    )
                    inletIdx = iHalo

                elif btype == "outlet":
                    fluidState["Pressure"][iHalo] = config.outletConditions()

                elif btype == "transparent":
                    # For transparent BCs, seed with a fraction of the inlet pressure
                    # just to get a smooth initial field; this gets overwritten later
                    # by the transparent BC call inside setBoundaryConditions().
                    fluidState["Pressure"][iHalo] = inletConditionsValues[0] / 10

            # Initialize static pressure linearly across the domain from the two edge values.
            fluidState["Pressure"] = np.interp(
                meshData["xMeshNodes"],
                [meshData["xMeshNodes"][0], meshData["xMeshNodes"][-1]],
                [fluidState["Pressure"][0],  fluidState["Pressure"][-1]],
            )

            if config.fluidModel() == "real":
                # Initialize density and energy assuming isentropic expansion from the inlet.
                inletEntropy = fluidModel.computeEntropy_p_rho(
                    fluidState["Pressure"][inletIdx], fluidState["Density"][inletIdx]
                )
                entropyField = np.full_like(fluidState["Pressure"], inletEntropy)
                fluidState["Density"] = fluidModel.computeDensity_p_s(
                    fluidState["Pressure"], entropyField
                )

            if config.fluidModel() == "ideal":
                # Initialize density assuming isentropic expansion from the inlet. 
                # for ideal fluid modelling, use the isentropic expansion equations for this. 
                fluidState["Density"] = fluidModel.computeDensityIsentropic_p1_p2_rho1(
                    fluidState["Pressure"][inletIdx], fluidState["Pressure"], fluidState["Density"][inletIdx]
                )

            # If the density field contains NaNs, the outlet pressure is likely too low
            # (e.g. because of the /10 seed for a transparent BC).  Bisect to find the
            # lowest pressure that still gives valid density, then re-initialize.
            if np.isnan(fluidState["Density"]).any():
                pressureBad  = fluidState["Pressure"][-1]
                pressureGood = fluidState["Pressure"][inletIdx]

                # Build a scalar density function that mirrors the happy-path dispatch,
                # so that the bisection works for both ideal and real fluid models.
                if config.fluidModel() == "real":
                    # Real fluid: density from (p, s).
                    def _densityFromPressure(p_scalar):
                        return fluidModel.computeDensity_p_s(p_scalar, inletEntropy)
                elif config.fluidModel() == "ideal":
                    # Both ideal and real: isentropic relation from inlet reference state.
                    rhoInlet = fluidState["Density"][inletIdx]
                    pInlet   = fluidState["Pressure"][inletIdx]
                    def _densityFromPressure(p_scalar):
                        return fluidModel.computeDensityIsentropic_p1_p2_rho1(
                            pInlet, p_scalar, rhoInlet
                        )

                if np.isnan(_densityFromPressure(pressureGood)):
                    raise RuntimeError(
                        "Cannot recover from NaN density field: inlet conditions also yield "
                        "NaN density (check inlet pressure / entropy / reference state)."
                    )

                for _ in range(50):  # bisection — 50 iterations is far more than enough
                    pressureMid  = 0.5 * (pressureBad + pressureGood)
                    densityMid   = _densityFromPressure(pressureMid)
                    if np.isnan(densityMid):
                        pressureBad  = pressureMid
                    else:
                        pressureGood = pressureMid

                pressureTest = pressureGood  # lowest converged pressure giving valid density

                print(
                    "Warning: The initialized density field contains NaN values. "
                    "Outlet BC is of transmissive kind; the outlet pressure imposed "
                    "(necessary for velocity initialization) was likely too low. "
                    f"Adjusting the outlet pressure to {pressureTest:.6e} Pa for initialization."
                )
                fluidState["Pressure"][-1] = pressureTest
                fluidState["Pressure"] = np.interp(
                    meshData["xMeshNodes"],
                    [meshData["xMeshNodes"][0], meshData["xMeshNodes"][-1]],
                    [fluidState["Pressure"][0],  fluidState["Pressure"][-1]],
                )

                # Re-initialize density using the same dispatch as the happy path.
                if config.fluidModel() == "real":
                    fluidState["Density"] = fluidModel.computeDensity_p_s(
                        fluidState["Pressure"], entropyField
                    )
                elif config.fluidModel() == "ideal":
                    fluidState["Density"] = fluidModel.computeDensityIsentropic_p1_p2_rho1(
                        fluidState["Pressure"][inletIdx], fluidState["Pressure"], rhoInlet
                    )

            fluidState["Energy"] = fluidModel.computeInternalEnergy_p_rho(
                fluidState["Pressure"], fluidState["Density"]
            )

            # Initialize velocity.
            if isStaticInlet:
                pass # velocity already initialized, was necessary for
                     # boundary condition specification, necessary for linear init.
            else:
                # Total inlet conditions: u_i = sqrt(2*(e_t - e_s)), where h_s is
                # evaluated from the local pressure and the inlet entropy.
                if inletConditionsVars == "ptTt":
                    totalTemperature = inletConditionsValues[1]
                    if config.fluidModel() == "real":
                        totalInternalEnergy = fluidModel.computeInternalEnergy_p_T(
                            fluidState["Pressure"][inletIdx], totalTemperature
                        )
                    else:
                        totalInternalEnergy = fluidModel.computeTotalInternalEnergy_Tt(totalTemperature)
                elif inletConditionsVars == "ptQt":
                    if config.fluidModel() == "real":
                        totalPressure, totalQuality = inletConditionsValues[:2]
                        totalInternalEnergy = fluidModel.computeInternalEnergy_p_Q(totalPressure, totalQuality)
                    else:
                        raise NotImplementedError(
                            "Total inlet conditions specified via (pt, Qt) are not supported "
                            "for ideal fluid models."
                        )
                        
                totalInternalEnergyField = np.full_like(fluidState["Pressure"], totalInternalEnergy)
                fluidState["Velocity"] = np.sqrt(
                    2 * (totalInternalEnergyField - fluidState["Energy"])
                )

            return fluidState

        
        def _imposeInitialConditionsNozzleUniform(config, meshData, fluidModel, fluidState):
            """
            Initialize fluid state variables uniformly across the nozzle domain from
            user-specified values in the configuration file.  Supports two specification
            modes:
            - (pressure, density, velocity)
            - (pressure, temperature, velocity)
            The internal energy is derived from whichever pair is provided.

            Arguments
            ---------
            config : Config
            meshData : dict
            fluidModel : FluidIdeal or FluidReal
            fluidState : dict  (modified in-place and returned)

            Returns
            -------
            fluidState : dict
            """
            p = config.initialPressure()
            v = config.initialVelocity()

            try:
                rho = config.initialDensity()
                T   = fluidModel.computeTemperature_p_rho(p, rho)
            except Exception:
                T   = config.initialTemperature()
                rho = fluidModel.computeDensity_p_T(p, T)

            e = fluidModel.computeInternalEnergy_p_rho(p, rho)

            for key, value in zip(
                ("Density", "Velocity", "Pressure", "Energy"),
                (rho,       v,          p,           e),
            ):
                fluidState[key][:] = value

            print(f"Uniform initial density    [kg/m3]: {rho:.6e}")
            print(f"Uniform initial velocity   [m/s]:   {v:.6e}")
            print(f"Uniform initial pressure   [Pa]:    {p:.6e}")
            print(f"Uniform initial temperature [K]:    {T:.6e}")
            print(f"Uniform initial energy     [J/kg]:  {e:.6e}")

            return fluidState

        # ------------------------------------------------------------------
        # Dispatch to the correct initialization helper.
        # ------------------------------------------------------------------
        deviceType = config.expansionDeviceType()
        fluidModelName = config.fluidModel().lower()
        bcs = [bc.lower() for bc in config.boundaryConditions()]

        if deviceType == "shocktube":
            fluidState = _imposeInitialConditionsShocktube(config, deviceGeometryData, meshData, fluidModel, fluidState)

        elif deviceType == "nozzle":
            # Linear initialization is only supported for real-fluid nozzle simulations
            # with (inlet, outlet) or (inlet, transparent) boundary conditions.
            linearBCPairs = [
                ("inlet", "outlet"),
                ("outlet", "inlet"),
                ("inlet", "transparent"),
                ("transparent", "inlet"),
            ]
            if tuple(bcs) in linearBCPairs:
                fluidState = _imposeInitialConditionsNozzleLinear(
                    config, meshData, fluidModel, fluidState
                )
            else:
                # For any other BC combination (e.g. reflective/periodic/transparent-
                # transparent), fall back to a user-specified uniform initial condition.
                fluidState = _imposeInitialConditionsNozzleUniform(
                    config, meshData, fluidModel, fluidState
                )

        return fluidState


    # =========================================================================
    #  Boundary conditions
    # =========================================================================

    def setBoundaryConditions(self, config, meshData, fluidModel, fluidState):
        """
        Evaluate and impose boundary conditions on the halo nodes of the fluid state
        arrays.  Implementation follows the ghost-node method described in Toro,
        "Riemann Solvers and Numerical Methods for Fluid Dynamics", 3rd ed., §6.3.3.

        Each halo node (index 0 for the left boundary, index -1 for the right boundary)
        is filled according to the boundary condition type specified in the configuration
        file.  The specific fill value depends on the type:

        - reflective   : mirror the neighbouring interior node, negating velocity.
        - transparent  : copy the neighbouring interior node unchanged (zero-gradient).
        - periodic     : copy from the opposite end of the physical domain.
        - inlet        : compute from total or static inlet conditions.
        - outlet       : impose a fixed back-pressure; fall back to transparent if supersonic.

        Arguments
        ---------
        config : Config
            The configuration object containing the boundary condition types and values.
        meshData : dict
            Mesh data dictionary produced by generateMesh().
        fluidModel : FluidIdeal or FluidReal
            An instance of the fluid model class, used for thermodynamic lookups.
        fluidState : dict
            The current fluid state (fluid state variables).  Modified in-place and
            returned so the calling code can keep a clean data-flow style.

        Returns
        -------
        fluidState : dict
            The fluid state dict with halo nodes updated according to the BCs.
        """
        bcLeft, bcRight = [bc.lower() for bc in config.boundaryConditions()]

        # Apply left boundary condition.
        if bcLeft == "reflective":
            fluidState = _applyReflectiveBC("left", fluidState)
        elif bcLeft == "transparent":
            fluidState = _applyTransparentBC("left", fluidState)
        elif bcLeft == "periodic":
            fluidState = _applyPeriodicBC("left", fluidState)
        elif bcLeft == "inlet":
            isTotalInlet = config.inletConditionsType().lower() == "total"
            inletConditionsValues = config.inletConditionsValues()
            if isTotalInlet:
                if 0.0 <= inletConditionsValues[1] <= 1.0:
                    inletConditionsVars = "ptQt"
                else:
                    inletConditionsVars = "ptTt"
            else:
                inletConditionsVars = None  # not needed for static inlet
            fluidState = _applyInletBC(
                0, 1, fluidModel, fluidState,
                isTotalInlet, inletConditionsVars, inletConditionsValues,
            )
        elif bcLeft == "outlet":
            fluidState = _applyOutletBC("left", 0, 1, config, fluidModel, fluidState)

        # Apply right boundary condition.
        if bcRight == "reflective":
            fluidState = _applyReflectiveBC("right", fluidState)
        elif bcRight == "transparent":
            fluidState = _applyTransparentBC("right", fluidState)
        elif bcRight == "periodic":
            fluidState = _applyPeriodicBC("right", fluidState)
        elif bcRight == "inlet":
            inletConditionsValues = config.inletConditionsValues()
            if isTotalInlet:
                if 0.0 <= inletConditionsValues[1] <= 1.0:
                    inletConditionsVars = "ptQt"
                else:
                    inletConditionsVars = "ptTt"
            else:
                inletConditionsVars = None  # not needed for static inlet
            fluidState = _applyInletBC(
                -1, -2, fluidModel, fluidState,
                isTotalInlet, inletConditionsVars, inletConditionsValues,
            )
        elif bcRight == "outlet":
            fluidState = _applyOutletBC("right", -1, -2, config, fluidModel, fluidState)

        return fluidState


    # =========================================================================
    #  Restart I/O
    # =========================================================================

    @staticmethod
    def extractRestartData(restartFilePath):
        """
        Extract restart data from a previously saved simulation step.  The output
        pickle file produced by saveResults() doubles as the restart file.

        Arguments
        ---------
        restartFilePath : str
            Path to the pickle restart file.

        Returns
        -------
        timeElapsed : float
            Physical time elapsed at the saved step.
        fluidStateRestart : dict
            Fluid state variable arrays at the saved step.
        configRestart : Config
            Configuration object stored alongside the results.
        iterationIndex : int
            Iteration counter at the saved step.
        """
        with open(restartFilePath, "rb") as fh:
            restartData = pickle.load(fh)

        timeElapsed        = restartData["time"]
        fluidStateRestart  = restartData["fluidState"]
        configRestart      = restartData["config"]
        iterationIndex     = restartData["iterationIdx"]

        return timeElapsed, fluidStateRestart, configRestart, iterationIndex


    # =========================================================================
    #  Output paths
    # =========================================================================

    @staticmethod
    def _prepareOutputPaths(config, meshData, restartFilePath):
        """
        Prepare and return the output directory path for simulation results.

        If continuing from a restart, results are written into the same directory
        as the restart file (no new folder is created).  For a clean start, a
        subdirectory is created under Results/; if overwrite is disabled and the
        directory already exists, a counter suffix is appended until a unique name
        is found.

        Arguments
        ---------
        config : Config
            The configuration object.
        meshData : dict
            Used only to read the node count for the directory name.
        restartFilePath : str or None
            Path to the restart file, or None for a clean start.

        Returns
        -------
        resultsPath : Path
            The (created) directory where results will be written.
        """
        if restartFilePath is not None:
            # Append new iteration files to the same folder as the restart file.
            return Path(restartFilePath).parent

        numNodes = meshData["numMeshNodes"] - 2  # subtract halo nodes for the label
        baseName = f"{config.resultsDirectoryName()}_NX_{numNodes}"
        resultsRoot = Path("Results")
        resultsRoot.mkdir(parents=True, exist_ok=True)
        resultsPath = resultsRoot / baseName

        if config.overwriteResults():
            if resultsPath.exists() and resultsPath.is_dir():
                shutil.rmtree(resultsPath)
        else:
            counter = 1
            candidate = resultsPath
            while candidate.exists():
                candidate = resultsRoot / f"{baseName}_{counter}"
                counter += 1
            resultsPath = candidate

        resultsPath.mkdir(parents=True, exist_ok=True)
        return resultsPath


    # =========================================================================
    # conservative <-> fluid state conversion
    # =========================================================================

    @staticmethod
    def _conservativesFromFluidState(fluidState, fluidModel):
        """
        Compute conservative variables from the fluid state and return them as a
        dictionary with keys 'u1', 'u2', 'u3'. Wrapper function to make pre-processing
        code more compact.

        Arguments
        ---------
        fluidState : dict
            Current fluid state variable arrays.
        fluidModel : FluidIdeal or FluidReal

        Returns
        -------
        conservativeState : dict
        """
        u1, u2, u3 = getConservativesFromFluidState(
            fluidState["Density"],
            fluidState["Velocity"],
            fluidState["Pressure"],
            fluidModel,
        )
        return {"u1": u1, "u2": u2, "u3": u3}


    @staticmethod
    def _fluidStateFromConservatives(conservativeState, fluidModel):
        """
        Compute fluid state variables from the conservative state and return them as a
        dictionary with keys 'Density', 'Velocity', 'Pressure', 'Energy'. Wrapper function
        to make pre-processing code more compact.

        Arguments
        ---------
        conservativeState : dict
        fluidModel : FluidIdeal or FluidReal

        Returns
        -------
        updates : dict of np.1darrays (interior nodes only)
        """
        rho, u, p, e = getFluidStateFromConservatives(
            conservativeState["u1"],
            conservativeState["u2"],
            conservativeState["u3"],
            fluidModel,
        )
        return {"Density": rho, "Velocity": u, "Pressure": p, "Energy": e}


    # =========================================================================
    #  Solver
    # =========================================================================
    def solve(self):
        """
        Advance the simulation in time using an explicit forward-Euler scheme.

        The numerical flux at each cell interface is computed using the scheme
        specified in the configuration file ('godunov', 'roe', 'roe_arabi', or
        'roe_vinokur').  Optional MUSCL reconstruction with a flux limiter can be
        activated to achieve second-order spatial accuracy.

        The loop runs until the physical time reaches timeMax, or until the solution
        has converged (all fluid state variables change by less than 1e-5 over 20
        consecutive iterations), in which case the timestep is advanced to timeMax
        in one shot to finalise the run.

        Returns
        -------
        None, but writes result files to self.resultsPath at intervals specified
        by writeInterval in the configuration file.
        """
        # Unpack all instance attributes up front.
        config              = self.config
        meshData            = self.meshData
        fluidModel          = self.fluidModel
        fluidState          = self.fluidState
        conservativeState   = self.conservativeState
        time                = self.time
        iterationIndex      = self.iterationIndex
        resultsPath         = self.resultsPath
        deviceGeometryData  = self.deviceGeometryData

        # Read solver settings from config.
        entropyFixActive      = config.entropyFixActiveBool()
        if entropyFixActive:
            entropyFixCoefficient = config.entropyFixCoefficient()
        else:
            entropyFixCoefficient = None
        advectionScheme       = config.numericalScheme()
        isMusclActive         = config.MUSCLReconstructionBool()
        writeInterval         = config.writeInterval()
        printResiduals        = config.printInfoResidualsBool()
        timeMax               = config.maxTime()
        cflMax                = config.CFLMax()
        expansionDeviceType   = config.expansionDeviceType()
        fluidModelType        = config.fluidModel()
        fluidLibrary          = config.fluidLibrary() if fluidModelType.lower() == "real" else None
        if isMusclActive:
            limiter = config.MUSCLReconstrFluxLimiter()
        else:
            limiter = None

        # plot nozzle geometry
        print("Plotting nozzle geometry...")
        print(expansionDeviceType)
        if expansionDeviceType == 'nozzle':
            print("Plotting nozzle geometry...")
            # nozzle x coordinates
            nozzleX = deviceGeometryData['deviceX']
            # nozzle area coordinates
            nozzleArea = deviceGeometryData['deviceArea']
            from matplotlib import pyplot as plt
            plt.figure()
            plt.plot(nozzleX, nozzleArea)
            plt.xlabel('x [m]')
            plt.ylabel('Area [m^2]')
            plt.title('Nozzle geometry')
            plt.grid()
            plt.show()

        print()
        print("=" * 80)
        print(" " * 33 + "START SOLVER")
        print("Numerical flux method: %s"  % advectionScheme)
        print("MUSCL reconstruction:  %s"  % isMusclActive)
        print("Entropy fix active:    %s"  % entropyFixActive)
        if fluidModelType.lower() == "real":
            print("Real Gas model, library: %s" % fluidLibrary)
        else:
            print("Ideal Gas model")
        if entropyFixActive:
            print("Entropy fix coefficient: %s" % entropyFixCoefficient)
        print("=" * 80)
        print()

        # Save the initial state (iteration 0, t = 0) for a clean start; a restart
        # already has its initial file so we skip this.
        if time == 0.0:
            saveResults(0, 0.0, meshData, fluidState, config, resultsPath)

        # Keep a copy of the previous step's fluid state variables for the convergence check.
        fluidStateOld   = copy.deepcopy(fluidState)
        convergenceHist = []
        convergedSimulation = False

        # -----------------------------------------
        # Iterative solution of governing equations
        # -----------------------------------------
        while time < timeMax:
            iterationIndex += 1

            # Compute the CFL-limited timestep and clip it so we land exactly on timeMax.
            dt      = computeTimeStep(fluidState, meshData, fluidModel, cflMax)
            dt      = min(dt, timeMax - time)
            newTime = time + dt

            # Compute residuals (finite-volume right-hand side).
            residuals = computeResiduals(
                fluidState, meshData, fluidModel, dt,
                advectionScheme, isMusclActive, limiter,
                entropyFixActive, entropyFixCoefficient,
                expansionDeviceType,
            )

            # Update conservative variables with the residuals, then recover fluid state variables.
            conservativeState, fluidState = updateSolution(
                conservativeState, fluidState, residuals, fluidModel
            )

            # Print progress.
            if printResiduals:
                _printInfoResiduals(iterationIndex, newTime, timeMax, residuals)
            else:
                print(
                    f"Iteration: {iterationIndex}, "
                    f"Progress in Time {(newTime / timeMax * 100):.3f} %"
                )

            # Periodic file output.
            if iterationIndex % writeInterval == 0:
                saveResults(
                    iterationIndex, newTime, meshData, fluidState, config, resultsPath
                )

            # Check for NaNs / Infs and abort with a diagnostic if found.
            checkSimulationStatus(fluidState, meshData, fluidModel, dt)

            # Re-impose boundary conditions on the halo nodes.
            fluidState = self.setBoundaryConditions(config, meshData, fluidModel, fluidState)
            # Keep conservative state consistent with the updated fluid state variables at the halos.
            conservativeState = self._conservativesFromFluidState(fluidState, fluidModel)

            # ------------------------------------------------------------------
            # Convergence check: if all fluid state variables have changed by less
            # than convergenceTolerance (relative) for 20 consecutive iterations,
            # jump straight to timeMax to finalise the run.
            # ------------------------------------------------------------------
            convergenceTolerance = 1e-5
            converged = all(
                np.max(
                    np.abs(fluidState[var] - fluidStateOld[var])
                    / (np.max(np.abs(fluidStateOld[var])) + 1e-300)
                ) < convergenceTolerance
                for var in ("Density", "Velocity", "Pressure", "Energy")
            )
            convergenceHist = convergenceHist + [True] if converged else []

            if len(convergenceHist) >= 20:
                # Force the loop to end at timeMax on the next iteration.
                dt = timeMax - time
                convergedSimulation = True

            # Advance physical time.
            time          = newTime
            fluidStateOld = copy.deepcopy(fluidState)

        # if nozzle simulation ended without converging due to time limit exceeding, inform the user
        # about this with a fair warning, suggesting to increase the time limit.
        if not convergedSimulation and expansionDeviceType == "nozzle":
            print("=" * 80)
            print("Warning: The simulation ended due to reaching the maximum time limit without convergence.")
            print("Consider increasing the maximum simulation time in the configuration file.")
            print("=" * 80)

        # Save the final state regardless of whether writeInterval aligns with it.
        saveResults(
            iterationIndex, time, meshData, fluidState, config, resultsPath
        )

        # Update Driver attributes so the object reflects the final simulation state.
        self.fluidState        = fluidState
        self.conservativeState = conservativeState
        self.time              = time
        self.iterationIndex    = iterationIndex

        print(" " * 34 + "END SOLVER")
        print("=" * 80)
        print(" " * 25 + "FINAL ASSEMBLY OF THE RESULTS")
        self.regroupSingleResults(resultsPath)
        print(" " * 34 + "END ASSEMBLER")
        print("=" * 80)


# =============================================================================
#  Additional helper functions
# =============================================================================


    def regroupSingleResults(self, filepath):
        # regrouping is only necessary when the results folder contains
        # files with filename RegEx: step*. 
        # Check for this
        files = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f)) and 'pik' in f]
        files = sorted(files)
        if not any(re.match(r'step_\d+\.pik', f) for f in files):
            print("No files with the expected naming convention found. No regrouping necessary.")
            return

        nTimes = len(files)
        solution = {}
        
        print("Regrouping all the results in a single file...")
        for iFile in range(len(files)):
            print(f"Reading File {iFile+1} of {len(files)}")
            with open(filepath / files[iFile], 'rb') as file:
                result = pickle.load(file)
                
                if iFile == 0:
                    nNodesVirtual = result['fluidState']['Pressure'].shape[0]
                    xNodesVirtual = result['xMeshNodes']
                    deviceAreaAtMeshNodes = result['deviceAreaAtMeshNodes']
                    config = result['config']
                    
                    timeVec = np.zeros(nTimes)
                    solution['Density'] = np.zeros((nNodesVirtual, nTimes))
                    solution['Velocity'] = np.zeros((nNodesVirtual, nTimes))
                    solution['Pressure'] = np.zeros((nNodesVirtual, nTimes))
                
                timeVec[iFile] = result['time']
                solution['Density'][:, iFile] = result['fluidState']['Density']
                solution['Velocity'][:, iFile] = result['fluidState']['Velocity']
                solution['Pressure'][:, iFile] = result['fluidState']['Pressure']
        
        globalOutput = {'xMeshNodes': xNodesVirtual, 
                        'deviceAreaAtMeshNodes': deviceAreaAtMeshNodes,
                        'Time': timeVec, 
                        'fluidState': solution, 
                        'config': config}
        
        print("Replacing all individual files with a single pickle (this could take a while) ...")
        shutil.rmtree(filepath)
        os.makedirs(filepath, exist_ok=True)
        with open(filepath / 'Results.pik', 'wb') as file:
            pickle.dump(globalOutput, file)
        print(f"Regrouped all the times in a single file: {filepath / 'Results.pik'}")

        


# -----------------------------------------------------------------------------
#  Boundary condition helpers
# -----------------------------------------------------------------------------

def _applyReflectiveBC(location, fluidState):
    """
    Fill the halo node to yield a reflective (solid-wall) boundary condition.
    All scalar quantities are mirrored from the adjacent interior node; the
    normal velocity component is negated to enforce zero mass flux through the wall.

    Reference: Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics",
    3rd ed., §6.3.3.

    Arguments
    ---------
    location : str
        'left' or 'right'.
    fluidState : dict
        Current fluid state arrays (modified in-place and returned).

    Returns
    -------
    fluidState : dict
    """
    iHalo, iInternal = (0, 1) if location == "left" else (-1, -2)
    fluidState["Density"][iHalo]  = fluidState["Density"][iInternal]
    fluidState["Velocity"][iHalo] = -fluidState["Velocity"][iInternal]   # sign flip
    fluidState["Pressure"][iHalo] = fluidState["Pressure"][iInternal]
    fluidState["Energy"][iHalo]   = fluidState["Energy"][iInternal]
    return fluidState


def _applyTransparentBC(location, fluidState):
    """
    Fill the halo node to yield a transparent (zero-gradient) boundary condition.
    All quantities are simply copied from the adjacent interior node, allowing
    waves to exit the domain without reflection.

    Reference: Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics",
    3rd ed., §6.3.3.

    Arguments
    ---------
    location : str
        'left' or 'right'.
    fluidState : dict
        Current fluid state arrays (modified in-place and returned).

    Returns
    -------
    fluidState : dict
    """
    iHalo, iInternal = (0, 1) if location == "left" else (-1, -2)
    for key in ("Density", "Velocity", "Pressure", "Energy"):
        fluidState[key][iHalo] = fluidState[key][iInternal]
    return fluidState


def _applyPeriodicBC(location, fluidState):
    """
    Fill the halo node to yield periodic boundary conditions.  The left halo
    receives the value from the last physical node; the right halo receives the
    value from the first physical node.

    Reference: "Formulation and Implementation of Inflow/Outflow Boundary
    Conditions to Simulate Propulsive Effects" / "Inflow/Outflow Boundary
    Conditions with Application to FUN3D".

    Arguments
    ---------
    location : str
        'left' or 'right'.
    fluidState : dict
        Current fluid state arrays (modified in-place and returned).

    Returns
    -------
    fluidState : dict
    """
    if location == "left":
        iHalo, iOpposite = 0, -2   # halo ← last physical node
    else:
        iHalo, iOpposite = -1, 1   # halo ← first physical node
    for key in ("Density", "Velocity", "Pressure", "Energy"):
        fluidState[key][iHalo] = fluidState[key][iOpposite]
    return fluidState


def _applyInletBC(iHalo, iInternal, fluidModel, fluidState,
                  isTotalInlet, inletConditionsVars, inletConditionsValues):
    """
    Fill the halo node to yield an inlet boundary condition.

    For total inlet conditions, the static pressure is extracted from the
    adjacent interior node and used together with the specified total conditions
    to recover density, velocity and internal energy via the fluid model.

    For static inlet conditions, density and energy are recovered directly from
    the specified static pressure and enthalpy; the velocity is taken from the
    adjacent interior node.

    Reference: "Formulation and Implementation of Inflow/Outflow Boundary
    Conditions to Simulate Propulsive Effects" / "Inflow/Outflow Boundary
    Conditions with Application to FUN3D".

    Arguments
    ---------
    iHalo : int
        Index of the halo node (0 for left, -1 for right).
    iInternal : int
        Index of the adjacent interior node (1 for left, -2 for right).
    fluidModel : FluidIdeal or FluidReal
    fluidState : dict
        Current fluid state arrays (modified in-place and returned).
    isTotalInlet : bool
        True if total inlet conditions are specified.
    inletConditionsVars : str or None
        'ptTt' or 'pQ' (only meaningful when isTotalInlet is True).
    inletConditionsValues : sequence
        The inlet condition values read from config.

    Returns
    -------
    fluidState : dict
    """
    if isTotalInlet:
        # The only information borrowed from the interior domain is the local
        # static pressure, which is used as the starting point for the iterative
        # inversion inside the fluid model.
        pressure = fluidState["Pressure"][iInternal]
        totalPressure = inletConditionsValues[0]

        # Guard against the static pressure being at or above total pressure,
        # which would break the isentropic relation inside the fluid model.
        if pressure >= totalPressure:
            pressure = 0.9999 * totalPressure

        if inletConditionsVars == "ptTt":
            totalTemperature = inletConditionsValues[1]
            massFlowDirection = inletConditionsValues[2]
            density, velocity, energy = fluidModel.computeInletQuantitiesTotal_pt_Tt(
                pressure, totalPressure, totalTemperature, massFlowDirection
            )
        elif inletConditionsVars == "ptQt":
            totalQuality     = inletConditionsValues[1]
            massFlowDirection = inletConditionsValues[2]
            density, velocity, energy = fluidModel.computeInletQuantitiesTotal_pt_Q(
                pressure, totalPressure, totalQuality, massFlowDirection
            )
        else:
            raise ValueError(
                f"Unknown inlet condition variable set '{inletConditionsVars}'. "
                "Must be 'ptTt' or 'ptQt'."
            )

    else: # static inlet
        # The only information borrowed from the interior domain is the local velocity. 
        velocity  = fluidState["Velocity"][iInternal]

        # same two options as the total conditions:
        if inletConditionsVars == "pT":
            pressure, totalTemperature = inletConditionsValues[:2]
            density, energy = fluidModel.computeInletQuantitiesStatic_p_T(pressure, totalTemperature)
        elif inletConditionsVars == "pQ":
            pressure, staticQuality = inletConditionsValues[:2]
            density, energy = fluidModel.computeInletQuantitiesStatic_p_Q(pressure, staticQuality)
        else:
            raise ValueError(
                f"Unknown inlet condition variable set '{inletConditionsVars}'. "
                "Must be 'ptTt' or 'pQ'."
            )

    fluidState["Density"][iHalo]  = density
    fluidState["Velocity"][iHalo] = velocity
    fluidState["Pressure"][iHalo] = pressure
    fluidState["Energy"][iHalo]   = energy

    return fluidState


def _applyOutletBC(location, iHalo, iInternal, config, fluidModel, fluidState):
    """
    Fill the halo node to yield a subsonic outlet boundary condition.

    For subsonic outflow (Mach < 1 at the adjacent interior node), the back
    pressure is fixed at the value specified in the configuration file while
    density and velocity are extrapolated from the interior.  For supersonic
    outflow (Mach ≥ 1), all information travels in the flow direction and the
    boundary is treated as transparent (zero-gradient).

    Reference: "Formulation and Implementation of Inflow/Outflow Boundary
    Conditions to Simulate Propulsive Effects" / "Inflow/Outflow Boundary
    Conditions with Application to FUN3D".

    Arguments
    ---------
    location : str
        'left' or 'right'.
    iHalo : int
        Index of the halo node.
    iInternal : int
        Index of the adjacent interior node.
    config : Config
    fluidModel : FluidIdeal or FluidReal
    fluidState : dict
        Current fluid state arrays (modified in-place and returned).

    Returns
    -------
    fluidState : dict
    """
    machOutlet = fluidModel.computeMach_u_p_rho(
        fluidState["Velocity"][iInternal],
        fluidState["Pressure"][iInternal],
        fluidState["Density"][iInternal],
    )

    if machOutlet < 1:
        # Subsonic: fix the back pressure, extrapolate everything else.
        pressure = config.outletConditions()
        density  = fluidState["Density"][iInternal]
        velocity = fluidState["Velocity"][iInternal]
        energy   = fluidModel.computeInternalEnergy_p_rho(pressure, density)

        fluidState["Density"][iHalo]  = density
        fluidState["Velocity"][iHalo] = velocity
        fluidState["Pressure"][iHalo] = pressure
        fluidState["Energy"][iHalo]   = energy
    else:
        # Supersonic: transparent (all characteristics point into the domain).
        fluidState = _applyTransparentBC(location, fluidState)

    return fluidState


# -----------------------------------------------------------------------------
#  Time stepping
# -----------------------------------------------------------------------------

def computeTimeStep(fluidState, meshData, fluidModel, cflMax):
    """
    Compute the maximum CFL-limited timestep over all interior nodes.

    The local stable timestep for a node i is dx_i / (|u_i| + a_i), where a_i
    is the local speed of sound.  The global timestep is the minimum over all
    interior nodes scaled by the user-specified CFL number.

    Arguments
    ---------
    fluidState : dict
        Current fluid state variable arrays (including halo nodes).
    meshData : dict
        Mesh data dictionary.
    fluidModel : FluidIdeal or FluidReal
    cflMax : float
        Maximum allowable CFL number.

    Returns
    -------
    dtMax : float
        The largest timestep that keeps every node below cflMax.
    """
    # Slice to interior nodes only (exclude the two halo nodes).
    velocity  = fluidState["Velocity"][1:-1]
    pressure  = fluidState["Pressure"][1:-1]
    density   = fluidState["Density"][1:-1]
    dx        = meshData["meshNodeSpacing"][1:-1]

    # Vectorised sound speed evaluation.
    soundSpeed = np.array([
        fluidModel.computeSoundSpeed_p_rho(pressure[i], density[i])
        for i in range(len(velocity))
    ])

    dtMax = np.min(dx * cflMax / (np.abs(velocity) + soundSpeed))
    return dtMax


# -----------------------------------------------------------------------------
#  Residual computation
# -----------------------------------------------------------------------------

def computeResiduals(fluidState, meshData, fluidModel, dt,
                     advectionScheme, isMusclActive, limiter,
                     entropyFixActive, entropyFixCoefficient,
                     expansionDeviceType):
    """
    Compute the finite-volume residual vector for all interior nodes.

    The residual for node i is:

        R_i = (dt / dx_i) * [(F_{i-1/2} - F_{i+1/2}) + S_i * dx_i]

    where F_{i±1/2} are the advection fluxes at the left and right cell
    interfaces and S_i is the quasi-1D area-variation source term (zero for
    constant-area geometries).

    Arguments
    ---------
    fluidState : dict
        Current fluid state variable arrays (including halo nodes).
    meshData : dict
        Mesh data dictionary.
    fluidModel : FluidIdeal or FluidReal
    dt : float
        Current timestep.
    advectionScheme : str
        One of 'godunov', 'roe', 'roe_arabi', 'roe_vinokur'.
    isMusclActive : bool
        Whether MUSCL second-order reconstruction is enabled.
    limiter : str
        Name of the flux limiter (e.g. 'van leer', 'min-mod', 'superbee').
    entropyFixActive : bool
    entropyFixCoefficient : float
    expansionDeviceType : str
        'nozzle' or 'shocktube'.

    Returns
    -------
    residuals : np.ndarray, shape (nPhysicalNodes, 3)
        The residual increment for each interior node and each conservation
        equation (mass, momentum, energy).
    """
    numMeshNodes  = meshData["numMeshNodes"]
    nPhysicalNodes = numMeshNodes - 2  # exclude the two halo nodes
    dx            = meshData["meshNodeSpacing"]
    xMeshNodes    = meshData["xMeshNodes"]

    # Compute advection fluxes on every internal interface (between node i and i+1
    # for i in [0, nPhysicalNodes], using halo nodes for the boundary interfaces).
    nFaces = nPhysicalNodes + 1
    flux   = np.zeros((nFaces, 3))
    for iFace in range(nFaces):
        iLeft  = iFace          # index into the full (halo-included) array
        iRight = iFace + 1
        flux[iFace, :] = computeFluxVector(
            iLeft, iRight, fluidState, meshData, fluidModel, dt,
            advectionScheme, isMusclActive, limiter,
            entropyFixActive, entropyFixCoefficient,
        )

    # Compute quasi-1D source terms for nozzle geometries; zero for constant area.
    if expansionDeviceType == "nozzle":
        source = computeSourceTerms(fluidState, meshData)
    else:
        source = np.zeros((numMeshNodes, 3))

    # Assemble the residual for each interior node.
    residuals = np.zeros((nPhysicalNodes, 3))
    for iDim in range(3):
        # flux[0:-1, iDim] is the flux at the left face of each interior node;
        # flux[1:,   iDim] is the flux at its right face.
        residuals[:, iDim] = (
            dt / dx[1:-1]
            * ((flux[:-1, iDim] - flux[1:, iDim]) + source[1:-1, iDim] * dx[1:-1])
        )

    return residuals


def computeFluxVector(iLeft, iRight, fluidState, meshData, fluidModel, dt,
                      advectionScheme, isMusclActive, limiter,
                      entropyFixActive, entropyFixCoefficient):
    """
    Compute the numerical flux vector at the interface between mesh nodes iLeft
    and iRight.

    If MUSCL reconstruction is active and the stencil is fully interior (at
    least one layer of real nodes on each side beyond the immediate neighbours),
    a higher-order reconstructed state is used.  Otherwise, the flux is computed
    from the piecewise-constant (first-order) left and right states.

    Arguments
    ---------
    iLeft, iRight : int
        Indices (into the full halo-included arrays) of the nodes on either side
        of the face.
    fluidState : dict
        Current fluid state variable arrays.
    meshData : dict
        Mesh data dictionary.
    fluidModel : FluidIdeal or FluidReal
    dt : float
        Current timestep (only needed by the Godunov scheme).
    advectionScheme : str
    isMusclActive : bool
    limiter : str
    entropyFixActive : bool
    entropyFixCoefficient : float

    Returns
    -------
    flux : np.ndarray, shape (3,)
        Numerical flux [F_mass, F_momentum, F_energy] at this face.
    """
    numMeshNodes = meshData["numMeshNodes"]

    # MUSCL reconstruction requires a two-cell stencil on each side of the face
    # (nodes iLeft-1 and iRight+1 must be valid array indices).
    musclApplicable = (
        isMusclActive
        and iLeft  >= 2
        and iRight <= numMeshNodes - 3
    )

    if musclApplicable:
        availableLimiters = ["van albada", "van leer", "min-mod", "superbee", "none"]
        if limiter not in availableLimiters:
            raise ValueError(
                f"Limiter '{limiter}' not recognized! Available ones are: {availableLimiters}"
            )
        rhoL, uL, pL, rhoR, uR, pR = computeMusclReconstruction(
            iLeft, iRight, fluidState, meshData, limiter
        )
    else:
        rhoL = fluidState["Density"][iLeft]
        rhoR = fluidState["Density"][iRight]
        uL   = fluidState["Velocity"][iLeft]
        uR   = fluidState["Velocity"][iRight]
        pL   = fluidState["Pressure"][iLeft]
        pR   = fluidState["Pressure"][iRight]

    # Dispatch to the chosen flux scheme.
    if advectionScheme.lower() == "godunov":
        if not isinstance(fluidModel, FluidIdeal):
            raise ValueError("Godunov scheme is available only for the ideal gas model.")
        dx_left  = meshData["meshNodeSpacing"][iLeft]
        dx_right = meshData["meshNodeSpacing"][iRight]
        nx, nt = 51, 51
        x = np.linspace(-dx_left / 2, dx_right / 2, nx)
        t = np.linspace(0, dt, nt)
        riem = RiemannProblem(x, t)
        riem.initializeState([rhoL, rhoR, uL, uR, pL, pR])
        riem.initializeSolutionArrays()
        riem.computeStarRegion()
        riem.solve(space_domain="interface", time_domain="global")
        rho, u, p = riem.getSolutionInTime()
        u1, u2, u3 = getConservativesFromFluidState(rho, u, p, fluidModel)
        u1AVG = np.mean(u1)
        u2AVG = np.mean(u2)
        u3AVG = np.mean(u3)
        flux = computeAdvectionFluxFromConservatives(u1AVG, u2AVG, u3AVG, fluidModel)

    elif advectionScheme.lower() == "roe":
        if isinstance(fluidModel, FluidReal):
            raise ValueError(
                "Basic Roe scheme is not available for the real gas model. "
                "Select 'roe_arabi' or 'roe_vinokur' instead."
            )
        roe  = AdvectionRoeBase(rhoL, rhoR, uL, uR, pL, pR, fluidModel)
        flux = roe.computeFlux(
            entropyFixActive=entropyFixActive, fixCoefficient=entropyFixCoefficient
        )

    elif advectionScheme.lower() == "roe_arabi":
        if isinstance(fluidModel, FluidIdeal):
            raise ValueError(
                "Roe_Arabi scheme is not available for the ideal gas model. "
                "Use the standard 'roe' scheme instead."
            )
        roe  = AdvectionRoeArabi(rhoL, rhoR, uL, uR, pL, pR, fluidModel)
        flux = roe.computeFlux(
            entropyFixActive=entropyFixActive, fixCoefficient=entropyFixCoefficient
        )

    elif advectionScheme.lower() == "roe_vinokur":
        roe = AdvectionRoeVinokur(rhoL, rhoR, uL, uR, pL, pR, fluidModel)
        roe.computeAveragedVariables()
        flux = roe.computeFlux(
            entropyFixActive=entropyFixActive, fixCoefficient=entropyFixCoefficient
        )

    else:
        raise ValueError(f"Unknown flux method '{advectionScheme}'.")

    return flux


def computeMusclReconstruction(iLeft, iRight, fluidState, meshData, limiter):
    """
    Perform MUSCL (Monotone Upstream-centred Schemes for Conservation Laws)
    reconstruction at the face between nodes iLeft and iRight.

    The reconstructed left and right interface states are:

        U_L* = U_L + 0.5 * psi_L * (U_R - U_L)
        U_R* = U_R - 0.5 * psi_R * (U_RP - U_R)

    where psi is the flux limiter evaluated from the smoothness indicator r,
    which measures the ratio of upstream to downstream gradients.

    Arguments
    ---------
    iLeft, iRight : int
        Node indices on either side of the face (into the full halo-included arrays).
    fluidState : dict
        Current fluid state variable arrays.
    meshData : dict
        Mesh data dictionary.
    limiter : str
        Name of the flux limiter.

    Returns
    -------
    rhoL, uL, pL, rhoR, uR, pR : float
        Reconstructed fluid states at the left and right sides of the face.
    """
    xMeshNodes = meshData["xMeshNodes"]

    # Four-point stencil: [iLeft-1, iLeft, iRight, iRight+1].
    U_lm = np.array([
        fluidState["Density"][iLeft - 1],
        fluidState["Velocity"][iLeft - 1],
        fluidState["Pressure"][iLeft - 1],
    ])
    U_l = np.array([
        fluidState["Density"][iLeft],
        fluidState["Velocity"][iLeft],
        fluidState["Pressure"][iLeft],
    ])
    U_r = np.array([
        fluidState["Density"][iRight],
        fluidState["Velocity"][iRight],
        fluidState["Pressure"][iRight],
    ])
    U_rp = np.array([
        fluidState["Density"][iRight + 1],
        fluidState["Velocity"][iRight + 1],
        fluidState["Pressure"][iRight + 1],
    ])

    # Cell spacings for the smoothness indicator computation.
    dx_lm_l  = xMeshNodes[iLeft]      - xMeshNodes[iLeft  - 1]
    dx_l_r   = xMeshNodes[iRight]     - xMeshNodes[iLeft]
    dx_r_rp  = xMeshNodes[iRight + 1] - xMeshNodes[iRight]

    # Smoothness indicators (ratio of consecutive gradients).
    r_left  = computeSmoothnessIndicators(U_lm, U_l,  U_r,  dx_lm_l, dx_l_r)
    r_right = computeSmoothnessIndicators(U_l,  U_r,  U_rp, dx_l_r,  dx_r_rp)

    # Flux limiters evaluated from the smoothness indicators.
    psi_left  = computeFluxLimiter(r_left,  limiter)
    psi_right = computeFluxLimiter(r_right, limiter)

    # Reconstruct left and right interface states.
    U_l_rec = U_l + 0.5 * psi_left  * (U_r  - U_l)
    U_r_rec = U_r - 0.5 * psi_right * (U_rp - U_r)

    return U_l_rec[0], U_l_rec[1], U_l_rec[2], U_r_rec[0], U_r_rec[1], U_r_rec[2]


def computeSmoothnessIndicators(U_left, U_central, U_right, dx_left, dx_right):
    """
    Compute the smoothness indicator vector r for use in a flux limiter.

    r_i = (dU_i / dx_left) / (dU_i / dx_right + epsilon)

    where epsilon is a small regularisation constant to avoid division by zero
    when the solution is locally flat.

    Arguments
    ---------
    U_left, U_central, U_right : np.ndarray, shape (3,)
        Fluid state variable vectors at the three stencil nodes.
    dx_left, dx_right : float
        Grid spacings on the left and right sides of the central node.

    Returns
    -------
    r : np.ndarray, shape (3,)
        Smoothness indicator for each fluid state variable.
    """
    r = ((U_central - U_left) / dx_left) / ((U_right - U_central) / dx_right + 1e-6)
    return r


def computeFluxLimiter(r_vec, limiter):
    """
    Evaluate the flux limiter function psi(r) for each component of r_vec.

    The limiter is applied component-wise to the smoothness indicator vector
    and returns the corresponding limiter value in [0, 2].

    Arguments
    ---------
    r_vec : np.ndarray, shape (3,)
        Smoothness indicator vector (one entry per fluid state variable).
    limiter : str
        Name of the limiter.  One of:
        - 'van albada' : smooth, differentiable
        - 'van leer'   : TVD, continuous
        - 'min-mod'    : most diffusive TVD limiter
        - 'superbee'   : least diffusive TVD limiter
        - 'none'       : no limiting (equivalent to psi = 1 everywhere)

    Returns
    -------
    psi : np.ndarray, shape (3,)
        Limiter values.
    """
    psi = np.zeros(3)
    for i, r in enumerate(r_vec):
        if limiter.lower() == "van albada":
            psi[i] = (r**2 + r) / (1 + r**2)
        elif limiter.lower() == "van leer":
            psi[i] = (r + np.abs(r)) / (1 + np.abs(r))
        elif limiter.lower() == "min-mod":
            psi[i] = np.maximum(0, np.minimum(1, r))
        elif limiter.lower() == "superbee":
            psi[i] = np.max([0, np.minimum(2 * r, 1), np.minimum(r, 2)])
        elif limiter.lower() == "none":
            psi[i] = 1
        else:
            raise ValueError(f"Limiter '{limiter}' not recognized!")
    return psi


# -----------------------------------------------------------------------------
#  Solution update
# -----------------------------------------------------------------------------

def updateSolution(conservativeState, fluidState, residuals, fluidModel):
    """
    Apply the residual increment to the conservative variables for all interior
    nodes, then recover the fluid state variables from the updated conservatives.

    Only the interior nodes (indices 1:-1) are updated; halo nodes are left
    unchanged here and will be overwritten by setBoundaryConditions().

    Arguments
    ---------
    conservativeState : dict
        Conservative variable arrays {'u1', 'u2', 'u3'} (modified in-place).
    fluidState : dict
        Fluid state variable arrays (modified in-place and returned).
    residuals : np.ndarray, shape (nPhysicalNodes, 3)
        Residual increment from computeResiduals().
    fluidModel : FluidIdeal or FluidReal

    Returns
    -------
    conservativeState : dict
    fluidState : dict
    """
    conservativeState["u1"][1:-1] += residuals[:, 0]
    conservativeState["u2"][1:-1] += residuals[:, 1]
    conservativeState["u3"][1:-1] += residuals[:, 2]

    # Recover fluid state variables from the updated conservatives (interior only).
    # function does not make use of the _fluidStateFromConservatives method for
    # simplicity's sake, as that would require some extra dictionary unpacking
    # steps. It uses the same getFluidStateFromConservatives math_utils function.
    rho, u, p, e = getFluidStateFromConservatives(
        conservativeState["u1"][1:-1],
        conservativeState["u2"][1:-1],
        conservativeState["u3"][1:-1],
        fluidModel,
    )
    fluidState["Density"][1:-1]  = rho
    fluidState["Velocity"][1:-1] = u
    fluidState["Pressure"][1:-1] = p
    fluidState["Energy"][1:-1]   = e

    return conservativeState, fluidState




# -----------------------------------------------------------------------------
#  Source terms
# -----------------------------------------------------------------------------

def computeSourceTerms(fluidState, meshData):
    """
    Compute quasi-1D source terms due to cross-sectional area variation along
    the nozzle.  The formulation is taken from Vimercati & Guardone, "On the
    numerical simulation of non-classical quasi-1D steady nozzle flows:
    Capturing sonic shocks".

    The source vector S at each node is:

        S_1 = -rho * u * (1/A) * dA/dx
        S_2 = -rho * u^2 * (1/A) * dA/dx
        S_3 = -u * (rho * E_tot + p) * (1/A) * dA/dx

    where E_tot = e + u^2/2 is the specific total energy.

    Arguments
    ---------
    fluidState : dict
        Current fluid state variable arrays (including halo nodes).
    meshData : dict
        Mesh data dictionary, providing deviceAreaAtMeshNodes and dAreaDx.

    Returns
    -------
    source : np.ndarray, shape (numMeshNodes, 3)
        Source term vectors at all mesh nodes (including halo nodes; the
        halo contributions are never added to the residual).
    """
    rho   = fluidState["Density"]
    u     = fluidState["Velocity"]
    p     = fluidState["Pressure"]
    e     = fluidState["Energy"]
    area  = meshData["deviceAreaAtMeshNodes"]
    dAdx  = meshData["dAreaDx"]

    totalEnergy = e + 0.5 * u**2   # specific total energy

    # Pre-compute the common geometric factor (avoids repeating the division).
    geomFactor = dAdx / area

    numMeshNodes = meshData["numMeshNodes"]
    source = np.zeros((numMeshNodes, 3))
    source[:, 0] = -rho * u                      * geomFactor
    source[:, 1] = -rho * u**2                   * geomFactor
    source[:, 2] = -u   * (rho * totalEnergy + p) * geomFactor

    return source


# -----------------------------------------------------------------------------
#  Diagnostics and output
# -----------------------------------------------------------------------------

def checkSimulationStatus(fluidState, meshData, fluidModel, dt):
    """
    Check for NaN or Inf values in the density and pressure fields and abort
    the simulation with a diagnostic message if any are found.

    The maximum local CFL number is printed to help identify the region where
    the solution has diverged, and a plot of the CFL distribution is displayed.

    Arguments
    ---------
    fluidState : dict
    meshData : dict
    fluidModel : FluidIdeal or FluidReal
    dt : float
        The timestep that was just used (needed for the CFL diagnostic).

    Returns
    -------
    None.  Calls sys.exit() if the simulation has diverged.
    """
    densityBad  = np.any(np.isnan(fluidState["Density"]))  or np.any(np.isinf(fluidState["Density"]))
    pressureBad = np.any(np.isnan(fluidState["Pressure"])) or np.any(np.isinf(fluidState["Pressure"]))

    if not (densityBad or pressureBad):
        return

    print()
    print("######################  SIMULATION DIVERGED ############################")
    print("NaNs or Infs detected in density or pressure. Simulation stopped.")

    # Compute the local CFL distribution using the state from the last valid step.
    cfl = _computeCFLField(fluidState, meshData, fluidModel, dt)
    print("Maximum CFL number found: %.3f"  % np.max(cfl))
    print("At location x: %.3f [m]"         % meshData["xMeshNodes"][1:-1][np.argmax(cfl)])
    print(
        "Visualize the plot to understand critical locations, "
        "and decrease CFL_MAX in the configuration file."
    )
    print("###############################  EXIT ##################################")
    print()

    plt.figure()
    plt.plot(meshData["xMeshNodes"][1:-1], cfl)
    plt.xlabel("x [m]")
    plt.ylabel("CFL [-]")
    plt.grid(alpha=0.3)
    plt.show()

    sys.exit()


def _computeCFLField(fluidState, meshData, fluidModel, dt):
    """
    Compute the local CFL number at every interior node.

    Arguments
    ---------
    fluidState : dict
    meshData : dict
    fluidModel : FluidIdeal or FluidReal
    dt : float

    Returns
    -------
    cfl : np.ndarray
        CFL number at each interior node.
    """
    pressure = fluidState["Pressure"][1:-1]
    density  = fluidState["Density"][1:-1]
    velocity = fluidState["Velocity"][1:-1]
    dx       = meshData["meshNodeSpacing"][1:-1]

    soundSpeed = np.array([
        fluidModel.computeSoundSpeed_p_rho(pressure[i], density[i])
        for i in range(len(velocity))
    ])

    cfl = (np.abs(velocity) + soundSpeed) * dt / dx
    return cfl


def _printInfoResiduals(iterationIndex, time, timeMax, residuals):
    """
    Print the L2 norm (in log10) of each equation's residual together with the
    current simulation progress.

    Arguments
    ---------
    iterationIndex : int
    time : float
        Physical time at the end of this iteration.
    timeMax : float
    residuals : np.ndarray, shape (nPhysicalNodes, 3)
    """
    nNodes = residuals.shape[0]
    res    = np.zeros(3)
    for iEq in range(3):
        normVal = np.linalg.norm(residuals[:, iEq]) / nNodes
        res[iEq] = np.log10(normVal) if normVal > 0 else -np.inf

    timeProgress = time / timeMax * 100
    print(
        "Iteration %i    Progress in Time %.3f%%    "
        "Residuals: %.6f, %.6f, %.6f"
        % (iterationIndex, timeProgress, res[0], res[1], res[2])
    )


def saveResults(iterationIndex, time, meshData, fluidState, config, resultsPath):
    """
    Serialize the current simulation state to a pickle file in resultsPath.

    The file contains enough information both for post-processing and for
    restarting the simulation from this point.  The filename encodes the
    iteration index so that multiple snapshots can coexist in the same directory.

    Arguments
    ---------
    iterationIndex : int
    time : float
        Physical time elapsed at this step.
    meshData : dict
    fluidState : dict
    config : Config
    resultsPath : Path

    Returns
    -------
    None.
    """
    filename = "step_%06i.pik" % iterationIndex
    fullPath = resultsPath / filename

    outputResults = {
        "time":                    time,
        "iterationIdx":            iterationIndex,
        "xMeshNodes":              meshData["xMeshNodes"],
        "deviceAreaAtMeshNodes":   meshData["deviceAreaAtMeshNodes"],
        "fluidState":              fluidState,
        "config":                  config,
    }

    with open(fullPath, "wb") as fh:
        pickle.dump(outputResults, fh)