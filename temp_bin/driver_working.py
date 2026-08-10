import os
import re
import pickle
import sys
import copy
import shutil

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pyshockflow.output import Output

from pyshockflow import Config
from pyshockflow import RiemannProblem
from pyshockflow import AdvectionRoeBase, AdvectionRoeArabi, AdvectionRoeVinokur
from pyshockflow import FluidIdeal, FluidReal

from pyshockflow.math_utils import (
    getPrimitivesFromConservatives,
    getConservativesFromPrimitives,
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
        fluidState         = self.setBoundaryConditions_old(config, meshData, fluidModel, fluidState)

        fluidState = {'Density': np.array([1.03475064, 0.96719449, 0.96719449, 0.96719449, 0.96719449,
       0.96719449, 0.96719449, 0.96719449, 0.96719449, 0.96719449,
       0.96719449, 0.96719449, 0.96719449, 0.96719449, 0.96719449,
       0.96719449, 0.96719449, 0.96719449, 0.96719449, 0.96719449,
       0.96719449, 0.96719449, 0.96719449, 0.96719449, 0.96719449,
       0.96719449, 0.96719449, 0.96719449, 0.96719449, 0.96719449,
       0.96719449, 0.96719449, 0.96719449, 0.96719449, 0.96719449,
       0.96719449, 0.96719449, 0.96719449, 0.96719449, 0.96719449,
       0.96719449, 0.96719449, 0.96719449, 0.96719449, 0.96719449,
       0.96719449, 0.96719449, 0.96719449, 0.96719449, 0.96719449,
       0.96719449, 0.96719449]), 'Velocity': np.array([194.42482782, 100.        , 100.        , 100.        ,
       100.        , 100.        , 100.        , 100.        ,
       100.        , 100.        , 100.        , 100.        ,
       100.        , 100.        , 100.        , 100.        ,
       100.        , 100.        , 100.        , 100.        ,
       100.        , 100.        , 100.        , 100.        ,
       100.        , 100.        , 100.        , 100.        ,
       100.        , 100.        , 100.        , 100.        ,
       100.        , 100.        , 100.        , 100.        ,
       100.        , 100.        , 100.        , 100.        ,
       100.        , 100.        , 100.        , 100.        ,
       100.        , 100.        , 100.        , 100.        ,
       100.        , 100.        , 100.        , 100.        ]), 'Pressure': np.array([80000., 80000., 80000., 80000., 80000., 80000., 80000., 80000.,
       80000., 80000., 80000., 80000., 80000., 80000., 80000., 80000.,
       80000., 80000., 80000., 80000., 80000., 80000., 80000., 80000.,
       80000., 80000., 80000., 80000., 80000., 80000., 80000., 80000.,
       80000., 80000., 80000., 80000., 80000., 80000., 80000., 80000.,
       80000., 80000., 80000., 80000., 80000., 80000., 80000., 80000.,
       80000., 80000., 80000., 45000.]), 'Energy': np.array([193283.2817238 , 206783.64375   , 206783.64375   , 206783.64375   ,
       206783.64375   , 206783.64375   , 206783.64375   , 206783.64375   ,
       206783.64375   , 206783.64375   , 206783.64375   , 206783.64375   ,
       206783.64375   , 206783.64375   , 206783.64375   , 206783.64375   ,
       206783.64375   , 206783.64375   , 206783.64375   , 206783.64375   ,
       206783.64375   , 206783.64375   , 206783.64375   , 206783.64375   ,
       206783.64375   , 206783.64375   , 206783.64375   , 206783.64375   ,
       206783.64375   , 206783.64375   , 206783.64375   , 206783.64375   ,
       206783.64375   , 206783.64375   , 206783.64375   , 206783.64375   ,
       206783.64375   , 206783.64375   , 206783.64375   , 206783.64375   ,
       206783.64375   , 206783.64375   , 206783.64375   , 206783.64375   ,
       206783.64375   , 206783.64375   , 206783.64375   , 206783.64375   ,
       206783.64375   , 206783.64375   , 206783.64375   , 116315.79960938])}

        # Conservative variables are derived from the primitives; initialise them
        # so that updateSolution() can operate on them from the very first step.
        conservativeState  = self._conservativesFromPrimitives(fluidState, fluidModel)

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
        fluidState         = self.setBoundaryConditions_old(config, meshData, fluidModel, fluidState)
        conservativeState  = self._conservativesFromPrimitives(fluidState, fluidModel)

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
        print("original device area", deviceGeometryData["deviceArea"])

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
            xMeshNodes = np.linspace(0, length, numMeshNodes)
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
            Initialize primitive variables on either side of the interface for shocktube
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
            Initialize primitive variables uniformly across the nozzle domain from
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

    def setBoundaryConditions_old(self, config, meshData, fluidModel, fluidState):
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
            The current fluid state (primitive variables).  Modified in-place and
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
            Primitive variable arrays at the saved step.
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
    # conservative ↔ primitive conversion
    # =========================================================================

    @staticmethod
    def _conservativesFromPrimitives(fluidState, fluidModel):
        """
        Compute conservative variables from the primitive state and return them as a
        dictionary with keys 'u1', 'u2', 'u3'.

        Arguments
        ---------
        fluidState : dict
            Current primitive variable arrays.
        fluidModel : FluidIdeal or FluidReal

        Returns
        -------
        conservativeState : dict
        """
        u1, u2, u3 = getConservativesFromPrimitives(
            fluidState["Density"],
            fluidState["Velocity"],
            fluidState["Pressure"],
            fluidModel,
        )
        return {"u1": u1, "u2": u2, "u3": u3}


    @staticmethod
    def _primitivesFromConservatives(conservativeState, fluidModel):
        """
        Compute primitive variables from the conservative state and return them as a
        dictionary with keys 'Density', 'Velocity', 'Pressure', 'Energy'.

        Arguments
        ---------
        conservativeState : dict
        fluidModel : FluidIdeal or FluidReal

        Returns
        -------
        updates : dict of np.1darrays (interior nodes only)
        """
        rho, u, p, e = getPrimitivesFromConservatives(
            conservativeState["u1"],
            conservativeState["u2"],
            conservativeState["u3"],
            fluidModel,
        )
        return {"Density": rho, "Velocity": u, "Pressure": p, "Energy": e}





    def setBoundaryConditions(self):
        """
        Set the boundary conditions to the halo nodes based on the type specified in the configuration file. The method calls the specific method for each type of boundary condition, 
        which updates the primitive variables in the halo nodes accordingly. BC specification according to the ghost node method (E. Toro Riemann Solvers and Numerical Methods for Fluid Dynamics
        Third Edition, section 6.3.3) In case of shock tube experiments, this code overwrites the left and right initialization. In case of nozzle flow, TODO: finish

        Arguments
        ---------
        None

        Returns
        -------
        None, but updates the solutionPrimitive attribute of the Driver class to update the value of the halo nodes based on the type specified in the configuration file.
        
        """
        if self.boundaryType[0].lower()=='reflective':
            self.setReflectiveBoundaryConditions('left')
        elif self.boundaryType[0].lower()=='transparent':
            self.setTransparentBoundaryConditions('left')
        elif self.boundaryType[0].lower()=='periodic':
            self.setPeriodicBoundaryConditions('left')
        elif self.boundaryType[0].lower()=='inlet':
            self.setInletBoundaryConditions('left')
        elif self.boundaryType[0].lower()=='outlet':
            self.setOutletBoundaryConditions('left')
        else:
            raise ValueError("Unknown boundary condition type on the left")
        
        if self.boundaryType[1].lower()=='reflective':
            self.setReflectiveBoundaryConditions('right')
        elif self.boundaryType[1].lower()=='transparent':
            self.setTransparentBoundaryConditions('right')
        elif self.boundaryType[1].lower()=='periodic':
            self.setPeriodicBoundaryConditions('right')
        elif self.boundaryType[1].lower()=='outlet':
            self.setOutletBoundaryConditions('right')
        elif self.boundaryType[1].lower()=='inlet':
            self.setInletBoundaryConditions('right')
        else:
            raise ValueError("Unknown boundary condition type on the right")
        
        # update also the conservative variable arrays based on what has been done on the primitive
        self.solutionConservative['u1'], self.solutionConservative['u2'], self.solutionConservative['u3'] = (getConservativesFromPrimitives(
            self.solutionPrimitive['Density'], self.solutionPrimitive['Velocity'], self.solutionPrimitive['Pressure'], self.fluid))



    def setReflectiveBoundaryConditions(self, location):
        """
        Set halo node values to yield reflective boundary conditions (E. Toro Riemann Solvers and Numerical Methods for Fluid Dynamics Third Edition, section 6.3.3)

        Arguments
        ---------
        location : str
            The location of the boundary condition, either 'left' or 'right'.

        Returns
        -------
        None, but updates the solutionPrimitive attribute of the Driver class to set the reflective boundary conditions to the halo nodes based on the location specified in the argument.
        """
        if location=='left':
            self.solutionPrimitive['Density'][0] = self.solutionPrimitive['Density'][1]
            self.solutionPrimitive['Velocity'][0] = -self.solutionPrimitive['Velocity'][1]
            self.solutionPrimitive['Pressure'][0] = self.solutionPrimitive['Pressure'][1]
            self.solutionPrimitive['Energy'][0] = self.solutionPrimitive['Energy'][1]
        elif location=='right':
            self.solutionPrimitive['Density'][-1] = self.solutionPrimitive['Density'][-2]
            self.solutionPrimitive['Velocity'][-1] = -self.solutionPrimitive['Velocity'][-2]
            self.solutionPrimitive['Pressure'][-1] = self.solutionPrimitive['Pressure'][-2]
            self.solutionPrimitive['Energy'][-1] = self.solutionPrimitive['Energy'][-2]
        else:
            raise ValueError('Unknown location specified')
            
    

    def setTransparentBoundaryConditions(self, location):
        """
        Set halo node values to yield transparent boundary conditions (E. Toro Riemann Solvers and Numerical Methods for Fluid Dynamics Third Edition, section 6.3.3)

        Arguments
        ---------
        location : str
            The location of the boundary condition, either 'left' or 'right'.

        Returns
        -------
        None, but updates the solutionPrimitive attribute of the Driver class to set the transparent boundary conditions to the halo nodes based on the location specified in the argument.
        """
        if location=='left':
            self.solutionPrimitive['Density'][0] = self.solutionPrimitive['Density'][1]
            self.solutionPrimitive['Velocity'][0] = self.solutionPrimitive['Velocity'][1]
            self.solutionPrimitive['Pressure'][0] = self.solutionPrimitive['Pressure'][1]
            self.solutionPrimitive['Energy'][0] = self.solutionPrimitive['Energy'][1]
        elif location=='right':
            self.solutionPrimitive['Density'][-1] = self.solutionPrimitive['Density'][-2]
            self.solutionPrimitive['Velocity'][-1] = self.solutionPrimitive['Velocity'][-2]
            self.solutionPrimitive['Pressure'][-1] = self.solutionPrimitive['Pressure'][-2]
            self.solutionPrimitive['Energy'][-1] = self.solutionPrimitive['Energy'][-2]
        else:
            raise ValueError('Unknown location specified')
        
        
        
    def setPeriodicBoundaryConditions(self, location):
        """
        Set halo node values to yield periodic boundary conditions. ("Formulation and Implementation of Inflow/Outflow Boundary Conditions to Simulate Propulsive Effects", or
        "Inflow/Outflow Boundary Conditions with Application to FUN3D")

        Arguments
        ---------
        location : str
            The location of the boundary condition, either 'left' or 'right'.
            
        Returns
        -------
        None, but updates the solutionPrimitive attribute of the Driver class to set the periodic boundary conditions to the halo nodes based on the location specified in the argument.
        """
        if location=='left':
            self.solutionPrimitive['Density'][0] = self.solutionPrimitive['Density'][-2]
            self.solutionPrimitive['Velocity'][0] = self.solutionPrimitive['Velocity'][-2]
            self.solutionPrimitive['Pressure'][0] = self.solutionPrimitive['Pressure'][-2]
            self.solutionPrimitive['Energy'][0] = self.solutionPrimitive['Energy'][-2]
        elif location=='right':
            self.solutionPrimitive['Density'][-1] = self.solutionPrimitive['Density'][1]
            self.solutionPrimitive['Velocity'][-1] = self.solutionPrimitive['Velocity'][1]
            self.solutionPrimitive['Pressure'][-1] = self.solutionPrimitive['Pressure'][1]
            self.solutionPrimitive['Energy'][-1] = self.solutionPrimitive['Energy'][1]
        else:
            raise ValueError('Unknown location specified')
    
    

    def setInletBoundaryConditions(self, location):
        """
        Set inlet boundary conditions. (see "Formulation and Implementation of Inflow/Outflow Boundary Conditions to Simulate Propulsive Effects", or
        "Inflow/Outflow Boundary Conditions with Application to FUN3D")

        Arguments
        ---------
        location : str
            The location of the boundary condition, either 'left' or 'right'.

        Returns
        -------
        None, but updates the solutionPrimitive attribute of the Driver class to set the inlet boundary conditions to the halo nodes based on the location specified in the argument.
        """
        # handle left and right extremities with the same code
        if location=='right':
            iHalo = -1
            iInternal = -2
        elif location=='left':
            iHalo = 0
            iInternal = 1
        else:
            raise ValueError('Unknown location specified')
        
        inletConditions = self.config.inletConditionsValues()    
        if self.config.inletConditionsType().lower()=="total":
            totalPressure = inletConditions[0]
            totalTemperature = inletConditions[1]
            direction = inletConditions[2]
            # static pressure is the only info taken from the domain 
            pressure = self.solutionPrimitive['Pressure'][iInternal]
            if pressure>=totalPressure: # avoid the problems that can cause
                pressure = 0.99*totalPressure 
            density, velocity, energy = self.fluid.computeInletQuantitiesTotal_pt_Tt(pressure, totalPressure, totalTemperature, direction)
        elif self.config.inletConditionsType().lower()=="static":
            if self.fluidModel=='ideal':
                raise ValueError('Static inlet conditions are only supported for the real fluid model')
            pressure = inletConditions[0]
            enthalpy = inletConditions[1]
            # get flow velocity from the domain
            velocity = self.solutionPrimitive['Velocity'][iInternal]
            density, energy = self.fluid.computeInletQuantitiesStatic(pressure, enthalpy)
        else:
            raise ValueError('Unknown or no inlet conditions type specified in the configuration file')
        self.solutionPrimitive['Density'][iHalo] = density
        self.solutionPrimitive['Velocity'][iHalo] = velocity
        self.solutionPrimitive['Pressure'][iHalo] = pressure
        self.solutionPrimitive['Energy'][iHalo] = energy
    
    

    def setOutletBoundaryConditions(self, location):
        """
        Set outlet boundary conditions. (see "Formulation and Implementation of Inflow/Outflow Boundary Conditions to Simulate Propulsive Effects", or
        "Inflow/Outflow Boundary Conditions with Application to FUN3D")

        Arguments
        ---------
        location : str
            The location of the boundary condition, either 'left' or 'right'.

        Returns
        -------
        None, but updates the solutionPrimitive attribute of the Driver class to set the outlet boundary conditions to the halo nodes based on the location specified in the argument.
        """
        # handle left and right extremities with the same code
        if location=='right':
            iHalo = -1
            iInternal = -2
        elif location=='left':
            iHalo = 0
            iInternal = 1
        else:
            raise ValueError('Unknown location specified')
        
        machOutlet = self.fluid.computeMach_u_p_rho(self.solutionPrimitive['Velocity'][iInternal], self.solutionPrimitive['Pressure'][iInternal], self.solutionPrimitive['Density'][iInternal])        
        if machOutlet<1:
            pressure = self.config.outletConditions() # the pressure is the information taken from outside
            velocity = self.solutionPrimitive['Velocity'][iInternal]
            density = self.solutionPrimitive['Density'][iInternal]
            energy = self.fluid.computeInternalEnergy_p_rho(pressure, density)        
            self.solutionPrimitive['Density'][iHalo] = density
            self.solutionPrimitive['Velocity'][iHalo] = velocity
            self.solutionPrimitive['Pressure'][iHalo] = pressure
            self.solutionPrimitive['Energy'][iHalo] = energy
        else:            
            self.setTransparentBoundaryConditions(location) # the boundary is equivalent to a transparent condition
            







    # =========================================================================
    #  Solver
    # =========================================================================

    def solve(self):
        """
        Solve the equations explicitly in time (forward Euler) using a certain advectionScheme (`Godunov`, `Roe`, `WAF`). high_order
        specifies if applying or not high order reconstruction with limiters. At the moment only type one is working -> simply
        impose high_order=True
        """
        self.entropyFixActive = self.config.entropyFixActiveBool()
        self.entropyFixCoefficient = 0.2
        advectionScheme = self.config.numericalScheme()
        isMusclActive = self.config.MUSCLReconstructionBool()
        writeInterval = self.config.writeInterval()
        printInfoResidualsBool = self.config.printInfoResidualsBool()

        self.config      
        self.meshData        
        self.deviceGeometryData 
        self.fluidModel      
        self.fluidState      
        self.conservativeState 
        self.resultsPath     
        self.time           
        self.iterationIndex  

        # define all self. attributes that will be used in the solver
        self.restartFilePath = None
        self.config = self.config
        self.topology = self.config.expansionDeviceType()
        self.fluidName = self.config.fluidName()
        self.fluidModel = self.config.fluidModel()
        self.gmma = self.config.fluidGamma()
        self.Rgas = self.config.gasRConstant()
        self.fluid = self.instantiateFluidModel(self.config)
        # self.pressureLeft = self.config.initialPressureLeft()
        # self.pressureRight = self.config.initialPressureRight()
        # self.temperatureLeft = self.config.initialTemperatureLeft()
        # self.temperatureRight = self.config.initialTemperatureRight()
        # self.densityLeft = self.config.initialDensityLeft()
        # self.densityRight = self.config.initialDensityRight()
        # self.velocityLeft = self.config.initialVelocityLeft()
        # self.velocityRight = self.config.initialVelocityRight()
        # self.energyLeft = self.fluid.computeInternalEnergy_p_rho(self.pressureLeft, self.densityLeft)
        # self.energyRight = self.fluid.computeInternalEnergy_p_rho(self.pressureRight, self.densityRight)
        self.length = self.deviceGeometryData['deviceLength']
        self.nNodes = self.meshData['numMeshNodes'] - 2
        self.xNodes = self.meshData['xMeshNodes'][1:-1]
        self.nNodesHalo = self.meshData['numMeshNodes']
        self.xNodesVirtual = self.meshData['xMeshNodes']
        self.dx = self.meshData['meshNodeSpacing']
        self.areaTube = self.meshData['deviceAreaAtMeshNodes']
        self.dAreaTube_dx = self.meshData['dAreaDx']
        self.cflMax = self.config.CFLMax()
        self.timeMax = self.config.maxTime()
        self.boundaryType = self.config.boundaryConditions()
        self.solutionNames = ['Density', 'Velocity', 'Pressure', 'Energy']
        self.solutionPrimitive = self.fluidState
        self.solutionConsNames = ['u1', 'u2', 'u3']
        self.solutionConservative = self.conservativeState


        for attr, value in vars(self).items():
            print(f"{attr}: {value}")




        
        print()
        print("="*80)
        print(" "*33 + "START SOLVER")
        print("Numerical flux method: %s" %(advectionScheme))
        print("MUSCL reconstruction: %s" %isMusclActive)
        print("Entropy fix active: %s" %self.entropyFixActive)
        if self.config.fluidModel()=='real':
            print("Real Gas model, library: %s" %self.config.fluidLibrary())
        else:
            print("Ideal Gas model")
        if self.entropyFixActive:
            print("Entropy fix coefficient: %s" %self.entropyFixCoefficient)
        print("="*80)
        print()

        # short aliases (shallow copy, will change throughout the iteration loop)
        primitiveOld = copy.deepcopy(self.solutionPrimitive)
        
        # # prepare output paths based on config specification
        # self._prepareOutputPaths(config, meshData, restartFilePath=None)

        # write the initial time to a results file (used both for post-processing and for restart)
        if self.restartFilePath is None:
            self.saveResults(it=0, time=0)
        
        if self.restartFilePath is not None:
            pass
        else:
            self.time = 0
            self.iterationIndex = 0

        # start convergence history
        convergence_hist = []        
        
        # main loop
        while self.time < self.timeMax:
            # perform iteration update
            self.iterationIndex += 1

            dt = self.computeTimeStep(self.solutionPrimitive)
            if self.time + dt > self.timeMax:
                dt = self.timeMax - self.time
            newTime = self.time + dt
            residuals = self.computeResiduals(self.solutionPrimitive, dt)
            self.updateSolution(residuals)
            
            if printInfoResidualsBool:
                self.printInfoResiduals(self.iterationIndex, newTime, residuals)        
            else:
                print(f"Iteration: {self.iterationIndex}, Progress in Time {((newTime)/self.timeMax * 100):.3f} %")
            
            if self.iterationIndex%writeInterval==0:
                self.saveResults(self.iterationIndex, newTime)

            self.checkSimulationStatus(dt)
            self.setBoundaryConditions()

            # convergence of primitive variables may carry differing time scales. Will simply check for convergence of all
            convergenceList = []
            convergenceTolerance = 1e-5
            for primitveVariable in self.solutionNames:
                # normalize the diff to get each variable on the same scale
                diff = np.abs(self.solutionPrimitive[primitveVariable] - primitiveOld[primitveVariable]) / np.max(np.abs(primitiveOld[primitveVariable]))
                if np.max(diff) < convergenceTolerance:
                    convergenceList.append(True)
                else:
                    convergenceList.append(False)
            if all(convergenceList):
                convergence_hist.append(self.iterationIndex)
            else: 
                convergence_hist = []
            if len(convergence_hist) >= 20:
                dt = self.timeMax - self.time

            # perform time update
            self.time += dt  
            primitiveOld = copy.deepcopy(self.solutionPrimitive)
        
        self.saveResults(self.iterationIndex, newTime)
            
        print(" "*34 + "END SOLVER")
        print("="*80)
        print(" "*25 + "FINAL ASSEMBLY OF THE RESULTS")
        output = Output(self.resultsPath)
        print(" "*34 + "END ASSEMBLER")
        print("="*80)

    
    
    def computeResiduals(self, primitives, dt):
        availableLimiters = ['van albada', 'van leer', 'min-mod', 'superbee', 'none']
        
        limiter = "van albada"
        if limiter not in availableLimiters:
            raise ValueError(f'Limiter not recognized! Available ones are: {availableLimiters}')
        
        advectionScheme = self.config.numericalScheme()
        MUSCL = self.config.MUSCLReconstructionBool()
        
        # compute advection fluxes on every internal interface
        flux = np.zeros((self.nNodes+1, 3))
        for iFace in range(flux.shape[0]):
            flux[iFace, :] = self.computeFluxVector(iFace, iFace+1, primitives, dt, advectionScheme, MUSCL, limiter)

        # compute the source terms
        if self.topology.lower()=='nozzle':
            source = self.computeSourceTerms(primitives)
        else:
            source = np.zeros((self.nNodesHalo,3))
        
        # assemble the full residual vector on every physical node
        residuals = np.zeros((self.nNodes,3))
        for iDim in range(3):
            residuals[:,iDim] = dt/self.dx[1:-1] * ((flux[0:-1, iDim] - flux[1:, iDim]) + source[1:-1, iDim]*self.dx[1:-1])

        return residuals



    def updateSolution(self, residuals):
        self.solutionConservative['u1'][1:-1] += residuals[:,0]
        self.solutionConservative['u2'][1:-1] += residuals[:,1]
        self.solutionConservative['u3'][1:-1] += residuals[:,2]
        self.updatePrimitivesFromConservatives()
    
    def updatePrimitivesFromConservatives(self):
        self.solutionPrimitive['Density'][1:-1], self.solutionPrimitive['Velocity'][1:-1], self.solutionPrimitive['Pressure'][1:-1], self.solutionPrimitive['Energy'][1:-1] = \
                getPrimitivesFromConservatives(self.solutionConservative['u1'][1:-1], self.solutionConservative['u2'][1:-1], self.solutionConservative['u3'][1:-1], self.fluid)
        


    def computeTimeStep(self, primitive):
        """
        Compute the maximum possible timestep given the pre-specified CFL number in the configuration file, and the spatial distribution of the physical + halo nodes, also specified in the configuration file. 
        The maximum CFL follows from numerical stability analysis of numerical governing equations (after applications of the chosen temporal and spatial discretization schemes)

        Arguments
        ---------
        primitive : dict of 2D np arrays, (space, time)
            The dictionary of primitive variables, containing the spatial distribution of density, velocity, pressure and energy at the current time step.

        Returns
        -------
        dtMax : float
            The maximum possible time step that can be taken at the current time step, given the spatial distribution of the primitive variables and the pre-specified CFL number in the configuration file.
        """
        velocity = primitive['Velocity'][1:-1]
        speedOfSound = np.zeros_like(velocity)
        for i in range(len(speedOfSound)):
            speedOfSound[i] = self.fluid.computeSoundSpeed_p_rho(primitive['Pressure'][i+1], primitive['Density'][i+1])
        print("speedOfSound:", speedOfSound)
        dtMax = np.min(self.dx[1:-1] * self.cflMax / (np.abs(velocity)+speedOfSound))
        # print("dtMax:", dtMax)
        return dtMax
    
    
    def saveResults(self, it, time):  
        """
        Save the results of the simulation at the current time step to a file in the results directory. The file is named according to the iteration index, and contains the time, iteration counter, 
        x coordinates of the nodes, area variation along the tube, primitive variables, fluid properties and configuration settings. The results are saved in a pickle file format.
        The results file is both used for post-processing and for restart. This is why seemingly unecessary information for restart (such as the area variation along the tube) is also present in the restart file. 

        Arguments
        ---------
        it : int
            The iteration index of the current time step.
        time : float
            The time elapsed at the current time step.

        Returns
        -------
        None, but saves the results of the simulation at the current time step to a file in the results directory, with the filename based on the iteration index.
        """  
        
        iterationName = 'step_%06i.pik' %(it)
        fullPath = self.resultsPath / iterationName
        outputResults = {'Time': time, 
                         'Iteration Counter': it, 
                         'X Coords': self.xNodesVirtual,
                         'Area Tube': self.areaTube,
                         'Primitive': self.solutionPrimitive,
                         'Configuration': self.config}
        with open(fullPath, 'wb') as file:
            pickle.dump(outputResults, file)
    
    
    def printInfoResiduals(self, iteration_idx, time, residuals):
        res = np.zeros(3)
        for iEq in range(3):
            res[iEq] = np.linalg.norm(residuals[:,iEq])/len(residuals[:,iEq])
            if res[iEq]!=0:
                res[iEq] = np.log10(res[iEq])
        timeProgress = time/self.timeMax * 100
        print('Iteration %i    Progress in Time %.3f%%    Residuals: %.6f, %.6f, %.6f' %(iteration_idx, timeProgress, res[0], res[1], res[2]))
    
    
    def computeSourceTerms(self, primitive):
        """compute source terms related to area variations along the tube due to a nozzle. Source terms taken from 'On the numerical simulation
        of non-classical quasi-1D steady nozzle flows: Capturing sonic shocks' by Vimercati and Guardone.

        Args:
            it (int): time step index

        Returns:
            np.ndarray: source terms arrays (nPoints, 3)
        """
        totalEnergy = primitive['Energy'][:] + 0.5*primitive['Velocity']**2
        source = np.zeros((self.nNodesHalo,3))
        print("your desired print statement", primitive["Density"].shape, primitive["Velocity"].shape, self.dAreaTube_dx.shape, self.areaTube.shape)
        source[:,0] = - primitive['Density'] * primitive['Velocity']*self.dAreaTube_dx/self.areaTube
        source[:,1] = - (primitive['Density'] * primitive['Velocity']**2)*self.dAreaTube_dx/self.areaTube
        source[:,2] = - primitive['Velocity'] *(primitive['Density']*totalEnergy + primitive['Pressure'])*self.dAreaTube_dx/self.areaTube
        return source
    
    
    def checkSimulationStatus(self, dt):
        """
        Check if nans or infs are detected and in that case stop the simulation and provide explanation
        """
        if np.any(np.isnan(self.solutionPrimitive['Density'])) or np.any(np.isinf(self.solutionPrimitive['Density'])) or \
            np.any(np.isnan(self.solutionPrimitive['Pressure'])) or np.any(np.isinf(self.solutionPrimitive['Pressure'])):
            print()
            print()
            print("######################  SIMULATION DIVERGED ############################")
            print('NaNs detected in density. Simulation stopped.')
            cfl = self.computeMaxCFL(dt) # use the previous time step to compute where the solution had CFL related problems
            print("Maximum CFL number found: %.3f" %(np.max(cfl)))
            print("At location x: %.3f [m]" %(self.xNodesVirtual[np.argmax(cfl)]))
            print("Visualize the plot to understand critical locations, and decrease CFL_MAX input setting.")
            print("###############################  EXIT ##################################")
            print()
            
            plt.figure()
            plt.plot(self.xNodes, cfl)
            plt.xlabel('x [m]')
            plt.ylabel('CFL [-]')
            plt.grid(alpha=.3)
            plt.show()
            sys.exit()
    
    
    def computeMaxCFL(self, dt):
        pressure = self.solutionPrimitive['Pressure'][1:-1]
        density = self.solutionPrimitive['Density'][1:-1]
        velocity = self.solutionPrimitive['Velocity'][1:-1]
        dx = self.dx[1:-1]
        soundSpeed = np.zeros_like(pressure)
        for i in range(len(soundSpeed)):
            soundSpeed = self.fluid.computeSoundSpeed_p_rho(pressure[i], density[i])
        cfl = (np.abs(velocity)+soundSpeed)*dt/dx
        return cfl
        

    def computeFluxVector(self, il, ir, primitive, dt, advectionScheme, MUSCL, limiter):
        """
        compute the flux vector at the interface between grid points `il` and `ir`, using a certain `advectionScheme`.
        """
        
        # flow reconstruction if high_order=True
        if (MUSCL and il>2 and ir<self.nNodesHalo-3):
            rhoL, uL, pL, rhoR, uR, pR = self.computeMusclReconstruction(il, ir, limiter)
        else:
            rhoL = primitive['Density'][il]
            rhoR = primitive['Density'][ir]
            uL = primitive['Velocity'][il]
            uR = primitive['Velocity'][ir]
            pL = primitive['Pressure'][il]
            pR = primitive['Pressure'][ir]            
        
        # flux calculation
        if advectionScheme.lower()=='godunov':
            if self.fluidModel!='ideal':
                raise ValueError('Godunov scheme is available only for ideal gas model')
            else:
                # Godunov flux calculation
                nx, nt = 51, 51 
                x = np.linspace(-self.dx[il]/2, self.dx[ir]/2, nx)
                t = np.linspace(0, dt, nt)
                riem = RiemannProblem(x, t)
                riem.initializeState([rhoL, rhoR, uL, uR, pL, pR])
                riem.initializeSolutionArrays()
                riem.computeStarRegion()
                riem.solve(space_domain='interface', time_domain='global') # compute Riemann solution only at x=0, but on all time instants
                rho, u, p = riem.getSolutionInTime()
                u1, u2, u3 = getConservativesFromPrimitives(rho, u, p, self.fluid)
                u1AVG, u2AVG, u3AVG = np.sum(u1)/len(u1), np.sum(u2)/len(u2), np.sum(u3)/len(u3)
                flux = computeAdvectionFluxFromConservatives(u1AVG, u2AVG, u3AVG, self.fluid) 
        elif advectionScheme.lower()=='roe':
            if self.fluidModel=='real':
                raise ValueError('Basic Roe scheme is not available for real gas model. Select Roe_Arabi or Roe_Vinokur, depending on the Roe Avg procedure that you want.')
            else:
                roe = AdvectionRoeBase(rhoL, rhoR, uL, uR, pL, pR, self.fluid)
                flux = roe.computeFlux(entropyFixActive=self.entropyFixActive, fixCoefficient=self.entropyFixCoefficient)
        elif advectionScheme.lower()=='roe_arabi':
            if self.fluidModel=='ideal':
                raise ValueError('Roe_Arabi scheme is not available for ideal gas model. Select Standard Roe scheme.')
            else:
                roe = AdvectionRoeArabi(rhoL, rhoR, uL, uR, pL, pR, self.fluid)
                flux = roe.computeFlux(entropyFixActive=self.entropyFixActive, fixCoefficient=self.entropyFixCoefficient)
        elif advectionScheme.lower()=='roe_vinokur':
                roe = AdvectionRoeVinokur(rhoL, rhoR, uL, uR, pL, pR, self.fluid)
                roe.computeAveragedVariables()
                flux = roe.computeFlux(entropyFixActive=self.entropyFixActive, fixCoefficient=self.entropyFixCoefficient)
        else:
            raise ValueError('Unknown flux method')
        
        return flux
    
    def computeMusclReconstruction(self, il, ir, limiter):
        """
        MUSCL reconstruction coupled with a certain limiter
        """
        # states left, left minus 1, right, right plus one
        U_lm = np.array([self.solutionPrimitive['Density'][il-1], self.solutionPrimitive['Velocity'][il-1], self.solutionPrimitive['Pressure'][il-1]])
        U_l = np.array([self.solutionPrimitive['Density'][il], self.solutionPrimitive['Velocity'][il], self.solutionPrimitive['Pressure'][il]])
        U_r = np.array([self.solutionPrimitive['Density'][ir], self.solutionPrimitive['Velocity'][ir], self.solutionPrimitive['Pressure'][ir]])
        U_rp = np.array([self.solutionPrimitive['Density'][ir+1], self.solutionPrimitive['Velocity'][ir+1], self.solutionPrimitive['Pressure'][ir+1]])
        
        dx_left_leftm = self.xNodes[il]-self.xNodes[il-1] # dx is always the same for now
        dx_right_left = self.xNodes[ir]-self.xNodes[il]
        dx_rightp_right = self.xNodes[ir+1]-self.xNodes[ir]
        
        # compute the smoothness indicators
        smoothnessLeft = self.computeSmoothnessIndicators(U_lm, U_l, U_r, dx_left_leftm, dx_right_left)
        smoothnessRight = self.computeSmoothnessIndicators(U_l, U_r, U_rp, dx_right_left, dx_rightp_right)
        
        # compute left and right flux limiters
        psi_left = self.computeFluxLimiter(smoothnessLeft, limiter)
        psi_right = self.computeFluxLimiter(smoothnessRight, limiter)
        
        # reconstruct left and right states
        U_l_rec = U_l+0.5*psi_left*(U_r-U_l)
        U_r_rec = U_r-0.5*psi_right*(U_rp-U_r)

        return U_l_rec[0], U_l_rec[1], U_l_rec[2], U_r_rec[0], U_r_rec[1], U_r_rec[2]


    def computeSmoothnessIndicators(self, U_left, U_central, U_right, dx_left, dx_right):
        """
        compute the array of smoothness indicators for the following flux limiter evaluation
        """
        rVector = ((U_central-U_left)/dx_left) / ((U_right-U_central)/dx_right + 1e-6)
        return rVector
    
    
    # def saveSolution(self):
    #     """
    #     Never used in Driver logic
    #     Save the full object as a pickle for later use
    #     """
    #     outputDirectoryName = self.config.getOutputDirectoryName()
    #     os.makedirs(outputDirectoryName, exist_ok=True)
    #     file_name = self.config.getOutputFileName()
    #     full_path = outputDirectoryName+'/'+file_name+'_NX_%i_TMAX_%.6f.pik' %(self.nNodes, self.timeMax)
    #     with open(full_path, 'wb') as file:
    #         pickle.dump(self, file)
    #     print('Pickle object with full solution saved to ' + full_path + ' !')


    def saveNodeSolutionToCSV(self, iNode, timeInstants, folder_name, file_name):
        """
        Save the array of fluid flow quantities (P,T,s,Mach,Gamma) from the solution to a CSV file.
        """
        filePath = folder_name + '/' + file_name + '.dat'

        pressure = self.solutionPrimitive['Pressure'][iNode, :]  # Extract the pressure data (1D array)
        density = self.solutionPrimitive['Density'][iNode, :]  # Extract the density data (1D array)
        temperature = self.fluid.computeTemperature_p_rho(pressure, density)
        entropy = self.fluid.computeEntropy_p_rho(pressure, density)
        fundDerGasDynamics = self.fluid.computeFunDerGamma_p_rho(pressure, density)
        compressibilityFactor = self.fluid.computeComprFactorZ_p_rho(pressure, density)

        with open(filePath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for value in range(len(timeInstants)):
                writer.writerow([timeInstants[value], pressure[value], temperature[value], density[value],
                                 entropy[value], fundDerGasDynamics[value], compressibilityFactor[value]])

        print(f"Fluid flow quantities (P,T,D,s,Gamma,Z) saved to {filePath}!")


    def computeFluxLimiter(self, r_vec, limiter):
        """
        compute the flux limiter functions.
        """
        psi = np.zeros(3)
        for i in range(len(r_vec)):
            r = r_vec[i]

            if limiter.lower() == 'van albada':
                psi[i] = (r**2+r)/(1+r**2)

            elif limiter.lower() == 'van leer':
                psi[i] = (r+np.abs(r))/(1+np.abs(r))

            elif limiter.lower() == 'min-mod':
                psi[i] = np.maximum(0, np.minimum(1, r))

            elif limiter.lower() == 'superbee':
                psi[i] = np.max(np.array([0, np.minimum(2 * r, 1), np.minimum(r, 2)]))

            elif limiter.lower() == 'none':
                psi[i] = 1 
            else:
                raise ValueError('Limiter not recognized!')
            
        return psi
    

    def readNozzleFile(self, xTube, filepath):
        nozzleData = np.loadtxt(filepath, skiprows=1, delimiter=',', dtype=float)
        nozzleX = nozzleData[:,0]
        nozzleArea = nozzleData[:,1]
        
        # Linear interpolation with external filling set to area Reference (=Tube area)
        interpolatedNozzleArea = np.interp(xTube, nozzleX, nozzleArea, left=nozzleData[0,1], right=nozzleData[-1,1])
    
        print(f"The reference tube area is: {nozzleData[0,1]:.6f} [m2].")
        print(f"The nozzle throat area is {interpolatedNozzleArea.min():.6f} [m2].")
        print(f"The nozzle maximum area is {interpolatedNozzleArea.max():.6f} [m2].")
        print(f"The area ratio between nozzle throat and exit section is {interpolatedNozzleArea.min()/interpolatedNozzleArea[-1]:.6f}.")
        print(f"The area ratio between nozzle throat and tube is {interpolatedNozzleArea.min()/nozzleData[0,1]:.6f}.")
        print(f"If this is not correct, modify the REFERENCE_AREA setting in the geometry section of the input file to the correct value for the tube area, or modify the nozzle csv file to be consistent with the tube area.")
        
        return interpolatedNozzleArea





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