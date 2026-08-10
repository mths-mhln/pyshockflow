import re
import os
import pickle
import csv
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
    getPrimitivesFromConservatives, 
    getConservativesFromPrimitives, 
    computeAdvectionFluxFromConservatives, 
    get_sign
)



class Driver:
    def __init__(self, configFilePath = None, restartFilePath=None):
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
            raise ValueError("Either a configuration file path or a restart file path must be provided.")
        if restartFilePath is not None: 
            # prepare Driver object to continue simulation from a previous step.
            # If user specifies a config file as well, the old simulation configuration
            # options, stored in the restartfile will be overwritten by the new ones. Be careful.
            self.prepareRestart(configFilePath, restartFilePath)
        if configFilePath is not None:
            # prepare Driver object to start a new simulation from scratch based 
            # on the configuration options provided by the user. 
            self.prepareCleanStart(configFilePath)

        

    def prepareCleanStart(self, configFilePath):
        """
        Initialize the Driver object for a new simulation from scratch based on the configuration 
        options provided by the user in the configuration file.

        Arguments
        ---------
        configFilePath : str
            The path to the configuration file, which contains the simulation parameters.

        Returns
        -------
        None, but sets the attributes of the Driver class based on the configuration file and 
        prepares the simulation for execution.
        """
        # Instantiate the Config object, capable of extracting information 
        # from the user-specified config file
        config = Config(configFilePath)

        # perform necessary preparations. For explanations of the computational 
        # procedures performed in each step, and the structure of the output, 
        # please refer to the method docstrings.
        deviceGeometryData: dict = self.extractDeviceGeometricalFeatures(config)
        meshData: dict = self.generateMesh(config, deviceGeometryData)
        fluidModel: object = self.instantiateFluidModel(config)
        fluidState: dict = self.initializeFluidStateArrays(config, meshData, fluidModel)
        fluidState: dict = self.imposeBoundaryConditions(config, meshData, fluidModel, fluidState)

        
        # set boundary conditions at the halo nodes 

        

        # specify numerical methods used for the solution of the governing equations on the mesh. 

        # prepare everything necessary for output directories. 





        # instantiate the config object. Depending on user input, 
        self.restartFilePath = restartFilePath
        if self.restartFilePath is not None:
            timeElapsed, solutionPrimitiveRestart, configRestart, iterationIndex = self.extractRestartData()
            print(f"Restarting simulation from file {self.restartFilePath} at iteration {iterationIndex} and time elapsed {timeElapsed:.6e} s")
            if configFilePath is None:
                self.config = configRestart
            else:
                # if both restart file and config file are provided, restart the simulation
                # by initializing the flow field with the data from the restart file, 
                # but proceed the solution with the new configuration in the config file. 
                self.config = Config(config_file = configFilePath)
        else:    
            self.config = Config(config_file = configFilePath)
        self.topology = self.config.Topology()
        self.fluidName = self.config.FluidName()
        self.fluidModel = self.config.FluidModel()
        
        if self.fluidModel.lower()=='ideal':
            self.gmma = self.config.FluidGamma()
            self.Rgas = self.config.GasRConstant()
            self.fluid = FluidIdeal(self.gmma,self.Rgas)
        elif self.fluidModel.lower()=='real':
            fluidLibrary = self.config.FluidLibrary()
            availFluidLibs = ['StanMix', 'GasMix', 'PCP-SAFT', 'RefProp', 'qPCP-SAFT', 'HOGC-PCP-SAFT',
                                'CoolProp', 'REFPROP', 'HEOS',
                                'Humid Air', 'Humid Air Mix', 'LuT', 'feos::HOGC-PCP-SAFT']
            if fluidLibrary not in availFluidLibs:
                raise ValueError(f"Invalid fluid library: {fluidLibrary}. Must be one of {availFluidLibs}")
            self.fluid = FluidReal(self.fluidName, fluidLibrary, self.config.PropertyExtractionMethod(), False)
        
        # geometry
        if self.topology.lower()=='nozzle':
            # length of the physical domain is not specified by the user as input. 
            # length must be extracted from the nozzle geometry file. 
            nozzleData = np.loadtxt(self.config.NozzleFilePath(), skiprows=1, delimiter=',', dtype=float)
            nozzleX = nozzleData[:,0]
            self.length = nozzleX[-1]-nozzleX[0]
        else:
            self.length = self.config.Length()
        self.nNodes = self.config.NumberOfPoints()
        xNodes = self.generatePhysicalGeometry(self.length, self.nNodes)
        self.generateVirtualGeometry(xNodes)
        
        # Time related information
        self.cflMax = self.config.CFLMax()
        self.timeMax = self.config.MaxTime()

        # deduce inlet condition type if total
        if self.config.InletConditionsType() == "total":
            inlet_conditions = self.config.InletConditionsValues()
            if 0.0 <= inlet_conditions[1] <= 1.0:
                self.inletConditionsVars = "pQ"
            else: 
                self.inletConditionsVars = "ptTt"
        
        # Boundary Conditions
        self.boundaryType = self.config.BoundaryConditions()    
        print("Boundary Conditions Left:                    %s" %self.boundaryType[0])
        print("Boundary Conditions Right:                   %s" %self.boundaryType[1])
        print("="*80)
        
        # Print info
        print("\n" + "=" * 80)
        print(" " * 25 + " WELCOME TO PYSHOCKTUBE ")
        print(" " * 18 + "Fluid Dynamics Simulation for Shock Tubes")
        print("=" * 80)
        print()  
        print("=" * 80)
        print(" "*32 + "SIMULATION DATA")
        print("Length of the domain [m]:                    %.6e" % self.length)
        print("Number of points:                            %i" % self.nNodes)
        print("Final time instant [s]:                      %.6e" % self.timeMax)
        print("Fluid name:                                  %s" % self.fluidName)
        print("Fluid treatment:                             %s" % self.fluidModel)
        if self.fluidModel.lower()=='ideal':
            print("Fluid cp/cv ratio [-]:                       %.6e" %self.gmma)
            print("Fluid gas constant [J/kgK]:                  %.6e" %self.Rgas)
        
        self.instantiatePrimitiveArrays()
        self.instantiateConservativeArrays()
        
        if self.restartFilePath is not None:
            # skip standard domain initialization, continue from restart domain.
            self.solutionPrimitive = solutionPrimitiveRestart
            self.time = timeElapsed
            self.iterationIndex = iterationIndex
        else:
            # standard domain initialization
            if self.topology.lower()=='nozzle':
                if self.fluidModel.lower()=='real' and ((self.boundaryType[0].lower() == "inlet" and self.boundaryType[1].lower() == "outlet") or (self.boundaryType[0].lower() == "outlet" and self.boundaryType[1].lower() == "inlet") or \
                   (self.boundaryType[0].lower() == "inlet" and self.boundaryType[1].lower() == "transparent") or (self.boundaryType[0].lower() == "transparent" and self.boundaryType[1].lower() == "inlet")):
                    # linear nozzle initialization is currently only supported for real fluids with inlet, outlet or transparent boundary conditions.
                    self.imposeInitialConditionsNozzleLinear()
                else:
                    # for other cases, initialization is done with constant primitive variable fields, just like for the shock tube case, hence
                    # the same functionality is used.
                    self.imposeInitialConditionsShocktube()
            else:
                # default topology.
                self.imposeInitialConditionsShocktube()
        self.setBoundaryConditions()
    




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
                - nozzleArea : np.1darray
                    The area of the nozzle cross-section at each point.

        Shock tube geometry:
            geometryData : dict
                A dictionary containing the extracted geometrical features.
                - shockTubeX : np.1darray
                    The physical length of the shock tube.
                - shockTubeArea : np.1darray
                    The area of the shock tube cross-section, assumed to be uniform.
                - shockTubeInterfaceLoc : float
                    The location of the interface between the high-pressure and low-pressure regions.
        """
        # extract the path to the CSV file containing the geometrical features from config.
        deviceGeometryFilePath = config.deviceGeometryFilePath()

        # instantiate geometryData dictionary
        geometryData = {}

        # Extract nozzle ordinates (physical distance along the nozzle) and coordinate
        # (area of the nozzle cross-section) and store those in the geometryData dictionary.
        nozzleData = np.loadtxt(deviceGeometryFilePath, skiprows=1, delimiter=',', dtype=float)
        geometryData['deviceX'] = nozzleData[:,0]
        geometryData['deviceArea'] = nozzleData[:,1]

        # according to the shock tube input data format requirements, the second data row
        # (disregarding the header) contains the interface location. 
        if config.expansionDeviceType() == "shocktube":
            geometryData['shockTubeInterfaceLoc'] = config.interfaceLocation()

        # generate some QoL information. 
        geometryData['deviceLength'] = geometryData['deviceX'][-1]-geometryData['deviceX'][0]

        return geometryData





    def generateMesh(self, config, deviceGeometryData):
        """
        Build the 1D mesh node positions along the shock tube. Generation is based on the expansion 
        device start and end coordinates (taken from the geometry data) and the mesh specifications 
        provided by the user in the configuration file. The mesh consists of physical domain and halo 
        nodes, at which boundary conditions will be imposed.

        If mesh refinement is enabled, the refinement zone uses a fixed number of uniformly-spaced 
        nodes (pointsRefinement). Outside the refinement zone, spacing transitions linearly from 
        dx_refined (at the refinement boundary) to dx_uniform (the spacing that would result from 
        placing `nodes` points uniformly over thefull domain), ensuring a smooth transition 
        without an abrupt jump.

        Arguments
        ---------
        config : Config
            The configuration object containing the mesh specifications.
        deviceGeometryData : dict
            A dictionary containing the geometrical features of the expansion device.
        """
        # Define an internal general-purpose helper function to build the mesh sections outside 
        # of the refined area if the user specifies refinement. 
        def _build_outside_section(start, end, n_points, dx_near, dx_far, direction):
            """
            Build a 1D mesh section with linearly varying spacing, transitioning smoothly
            from dx_near (at the refinement boundary) to dx_far (at the domain edge).

            Arguments
            ---------
            start, end : float
                endpoints of this section
            n_points : int
                number of nodes including both endpoints
            dx_near : float
                spacing at the refinement boundary
            dx_far : float
                spacing at the domain edge
            direction : str
                'upstream' (refinement on the right) or 'downstream' (refinement on the left)

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

            if direction == 'upstream':
                # Refinement is on the right: build right-to-left from the refinement boundary,
                # then reverse so coords run left-to-right.
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


        # instantiate meshData dictionary
        meshData = {}

        # Extract length of the expansion device computational domain and number 
        # of mesh nodes the user wants to place in it.
        length = deviceGeometryData['deviceLength']
        numMeshNodes = config.numberOfMeshNodes()

        # Check if mesh refinement is enabled in the configuration file. 
        # If not, generate a uniformally spaced mesh with numMeshNodes.
        isMeshRefined = config.meshRefinementBool()

        if not isMeshRefined:
            return np.linspace(0, length, numMeshNodes)

        refinementCoords = config.refinementBoundaries()
        print("Mesh is refined between the two boundaries [m]: ", refinementCoords)

        numMeshNodesRef = config.numberOfRefMeshNodes()
        x0_ref, x1_ref = refinementCoords

        # dx_refined: uniform spacing inside the refinement zone, used as the fine
        # anchor for the linear interpolation in the outside sections.
        dx_refined = (x1_ref - x0_ref) / numMeshNodesRef

        # dx_uniform: spacing of a hypothetical uniform mesh with `nodes` points over
        # the full domain, used as the coarse anchor for the linear interpolation.
        dx_uniform = length / (numMeshNodes - 1)

        # Build the refinement zone; endpoints are shared with the outside sections.
        xRefinement = np.linspace(x0_ref, x1_ref, numMeshNodesRef + 1)

        lengthUpstream   = x0_ref
        lengthDownstream = length - x1_ref
        lengthTotal      = lengthUpstream + lengthDownstream

        has_upstream   = lengthUpstream   > 0
        has_downstream = lengthDownstream > 0

        if has_upstream and has_downstream:
            # Split `nodes` proportionally by section length.
            n_upstream   = max(2, round(numMeshNodes * lengthUpstream   / lengthTotal))
            n_downstream = max(2, round(numMeshNodes * lengthDownstream / lengthTotal))

            xUpstream   = _build_outside_section(
                0, x0_ref, n_upstream, dx_refined, dx_uniform, 'upstream'
                )
            xDownstream = _build_outside_section(
                x1_ref, length, n_downstream, dx_refined, dx_uniform, 'downstream'
                )

            # Drop the last node of xUpstream and xRefinement to avoid duplicating
            # the shared boundary nodes at x0_ref and x1_ref.
            xMeshNodes = np.concatenate((xUpstream[:-1], xRefinement[:-1], xDownstream))

        elif has_upstream:
            # Refinement ends at the right edge of the domain.
            xUpstream = _build_outside_section(
                0, x0_ref, max(2, numMeshNodes), dx_refined, dx_uniform, 'upstream'
                )
            xMeshNodes = np.concatenate((xUpstream[:-1], xRefinement))

        elif has_downstream:
            # Refinement starts at the left edge of the domain.
            xDownstream = _build_outside_section(
                x1_ref, length, max(2, numMeshNodes), dx_refined, dx_uniform, 'downstream'
                )
            xMeshNodes = np.concatenate((xRefinement[:-1], xDownstream))

        else:
            raise ValueError(
                "The refinement zone covers the entire domain. "
                "Nothing is left to mesh outside it."
            )

        # mesh for the physical domain has been built, add halo nodes to the mesh. 
        # simply add a node at the left and right of the physical domain, with the 
        # same spacing as the first and last physical nodes have wrt their neighbors.
        xHaloLeft = xMeshNodes[0] - (xMeshNodes[1]-xMeshNodes[0])
        xHaloRight = xMeshNodes[-1] + (xMeshNodes[-1]-xMeshNodes[-2])
        xMeshNodes = np.concatenate(([xHaloLeft], xMeshNodes, [xHaloRight]))

        # save physical positions of mesh nodes along the Expansion Device 
        # in the meshData dictionary.
        meshData['xMeshNodes'] = xMeshNodes

        # compute some QoL information about the mesh, necessary for simplifying future
        # computational steps, and store it in the meshData dictionary.

        # total number of meshnodes
        meshData['numMeshNodes'] = len(xMeshNodes)
        
        # meshNodeSpacing: the physical width of each mesh cell. The width is computed
        # using second order accurate central differences based on the physical distance
        # of the mesh node to it's neighbors for the interior nodes and either first or 
        # second order accurate one-sides (forward or backwards) differences at the boundary 
        # nodes (physical mesh extremities and halo nodes). 
        meshData['meshNodeSpacing'] = np.gradient(xMeshNodes)

        # interpolate the device cross-sectional area variation at the mesh node locations. 
        # for this, the cross-sectional area variation stored in the deviceGeometryData 
        # dictionary is used. left and right specifications are necessary to ensure the 
        # geometry curvatur is not extrapolated to the ghost nodes, addiding non-existant
        # area variation to the geometry. Keep area constant outside the physical domain.
        meshData['deviceAreaAtMeshNodes'] = np.interp(
             xMeshNodes, deviceGeometryData['deviceX'], deviceGeometryData['deviceArea'],
             left = deviceGeometryData['deviceArea'][0], right = deviceGeometryData['deviceArea'][-1]
             )
        
        return meshData




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
            An instance of the fluid model class, capable of computing thermodynamic properties.
        """
        fluidName = config.FluidName()
        fluidModel = config.FluidModel()

        if fluidModel.lower() == 'ideal':
            gmma = config.FluidGamma()
            Rgas = config.GasRConstant()
            return FluidIdeal(gmma, Rgas)
        
        elif fluidModel.lower() == 'real':
            fluidLibrary = config.FluidLibrary()
            return FluidReal(fluidName, fluidLibrary, config.propertyExtractionMethod(), False)




    def initializeFluidStateArrays(self, config, deviceGeometryData, meshData, fluidModel):
        """
        Initialize fluid state arrays at the meshNodes. The fluid state arrays comprise four
        variables of interest: density, velocity, pressure and energy.

        For nozzle geometries, the fluid thermodynamic state is initialized in different ways, 
        depending on the boundary conditions specified by the user. If inlet and outlet, 
        or inlet and transparent boundary conditions are specified, the fluid thermodynamic
        state is initialized linearly between the boundary conditions if not, the state assumes
        the boundary conditions all throughout the geometry.

        For shocktube geometries, the fluid thermodynamic state is initialized based on user-
        -specified initial conditions, specifying the themrodyanamic state of the working fluid
        on each side of the interface. Conditions are assumed to be uniform on each side of the interface. 
        """
        # Instantiate fluid state dict and arrays (of similar )
        fluidState = {}
        fluidStateKeys = ['Density', 'Velocity', 'Pressure', 'Energy']
        for key in fluidStateKeys:
            fluidState[key] = np.zeros(meshData['numMeshNodes'])

        # helper functions to impose initial conditions for nozzle and shocktube geometries.
        def imposeInitialConditionsShocktube(self, config, meshData, deviceGeometryData, fluidModel, fluidState):
            """
            Initialize primitive variables on either side of the interface for shocktube experiments.
            Thermodynamic state is specified via (p, rho) if density is provided in the config, 
            otherwise via (p, T). The interface is imposed through copyInitialState.

            Arguments
            ---------
            config: Config
                The configuration object containing the simulation parameters.
            fluidModel: FluidIdeal or FluidReal
                An instance of the fluid model class, capable of computing thermodynamic properties.

            Returns
            -------
            None, but sets the solutionPrimitive attribute to the initial conditions.
            """
            # extract initial conditions for the shock tube experiment specified
            # by the user. Inputs are required to be either (p, v, rho) or (p, v, T).
            pL, pR = config.pressureLeft(), config.pressureRight()
            vL, vR = config.velocityLeft(), config.velocityRight()

            # Use density if provided, otherwise temperature. Compute the other from
            # the specified one, together with the pressure.
            try:
                rhoL, rhoR = config.densityLeft(), config.densityRight()
                TL = fluidModel.computeTemperature_p_rho(pL, rhoL)
                TR = fluidModel.computeTemperature_p_rho(pR, rhoR)
            except:
                TL, TR = config.temperatureLeft(), config.temperatureRight()
                rhoL = fluidModel.computeDensity_p_T(pL, TL)
                rhoR = fluidModel.computeDensity_p_T(pR, TR)

            # compute static internal energy form thermodynamic state.
            eL = fluidModel.computeInternalEnergy_p_rho(pL, rhoL)
            eR = fluidModel.computeInternalEnergy_p_rho(pR, rhoR)

            # group in dictionary to simplify computational operations.
            initialConditions = {
                'Density':  (rhoL, rhoR),
                'Velocity': (vL,   vR),
                'Pressure': (pL,   pR),
                'Energy':   (eL,   eR),
            }

            for key, (valueL, valueR) in initialConditions.items():
                xInterface = deviceGeometryData['shockTubeInterfaceLoc']
                fluidState[key] = np.where(
                    meshData["xMeshNodes"] <= xInterface, valueL, valueR
                )
            
            print(f"Initial L/R density values [kg/m3]:  ({rhoL:.6e}, {rhoR:.6e})")
            print(f"Initial L/R velocity values [m/s]:   ({vL:.6e}, {vR:.6e})")
            print(f"Initial L/R pressure values [Pa      ({pL:.6e}, {pR:.6e})")
            print(f"Initial L/R temperature values [K]   ({TL:.6e}, {TR:.6e})")
            print(f"Initial L/R energy values [J/kg      ({eL:.6e}, {eR:.6e})")

            return fluidState

        
        def imposeInitialConditionsNozzleLinear(self, meshData, fluidModel, fluidState):
            """
            Method reserved for nozzle flow experiments. Following the advice of 
            Cioffi et al. (2025) A Hyperbolic One-Dimensional Model for Two-Phase
            Flows in Converging-Diverging Nozzles, the flow field is initialized 
            in linear fashion from inlet to outlet.
    
            Arguments
            ---------
            None
    
            Returns
            -------
            None, but sets the solutionPrimitive attribute of the Driver class 
            to the initial conditions based on the left and right values provided 
            in the configuration file, and prints these values to the terminal.
            """
            isTotalInlet = config.inletConditionsType() == "total"
            inletIdx = None  # idx (0 or -1) of whichever side turns out to be the inlet
    
            # Impose boundary conditions at each edge of the domain, tracking which 
            # edge is the inlet.
            for iHalo, side, iInternal in [(0, 'left', 1), (-1, 'right', -2)]:
                btype = config.boundaryConditions()[0 if iHalo == 0 else 1]
    
                if btype == 'inlet':
                    if isTotalInlet:
                        # setInletBoundaryConditions needs a static pressure guess for 
                        # its iteration to converge; seed the neighboring node with the 
                        # (total) inlet pressure as a starting value.
                        fluidState['Pressure'][iInternal] = config.inletConditionsValues()[0]
                    fluidState = self.setInletBoundaryConditions(side, fluidState)
                    inletIdx = iHalo
    
                elif btype == 'outlet':
                    fluidState['Pressure'][iHalo] = config.outletConditions()
    
                elif btype == 'transparent':
                    # For transparent BCs, seed with a fraction of the inlet pressure just to 
                    # get a smooth initial field; this gets overwritten later by the transparent 
                    # BC method in driver.py.
                    fluidState['Pressure'][iHalo] = config.inletConditionsValues()[0] / 10
    
            # Initialize static pressure linearly across the domain from the two edge values.
            fluidState["Pressure"] = np.interp(
                meshData['xMeshNodes'],
                [meshData['xMeshNodes'][0], meshData['xMeshNodes'][-1]],
                [fluidState['Pressure'][0], fluidState['Pressure'][-1]]
            )
    
            # Initialize density and energy assuming isentropic expansion from the inlet.
            inletEntropy = fluidModel.computeEntropy_p_rho(
                fluidState['Pressure'][inletIdx], fluidState['Density'][inletIdx]
            )
            entropyField = np.full_like(fluidState['Pressure'], inletEntropy)
            fluidState["Density"] = fluidModel.computeDensity_p_s(fluidState["Pressure"], entropyField)
            # if corresponding density field contains NaN's, evaluate the density for the 
            # outlet pressure value and inlet entropy capture the stdout message, if it 
            # contains, anywhere, "below minimum", the outlet pressure is likely too low, 
            # This is a likely case for transparent outlet where the division by 10 leads 
            # to too low outlet pressure. If the boundary value is indeed of transparent 
            # kind, get the lowest pressure which does not lead to NaN density, and 
            # re-initialize the presure distribution to that value, rather than to division
            # by 10. But only for if BC is of the transparent kind. The user will be warned 
            # that this happened for debugging reasons. 
            if np.isnan(fluidState["Density"]).any():
                # find the lowest pressure that does not lead to NaN density for the given 
                # inlet entropy. The outlet pressure (pressureTest) is known-bad (NaN density); 
                # the inlet pressure is known-good, so bisect between them to converge on 
                # the threshold pressure.
                pressureBad = fluidState['Pressure'][-1]
                pressureGood = fluidState['Pressure'][inletIdx]
    
                if np.isnan(fluidModel.computeDensity_p_s(pressureGood, inletEntropy)):
                    raise RuntimeError(
                        "Cannot recover from NaN density field: inlet pressure also yields"
                        " NaN density for the given inlet entropy."
                    )
    
                for _ in range(50):  # bisection, ~50 iters is far more than enough for float precision
                    pressureMid = 0.5 * (pressureBad + pressureGood)
                    densityMid = fluidModel.computeDensity_p_s(pressureMid, inletEntropy)
                    if np.isnan(densityMid):
                        pressureBad = pressureMid
                    else:
                        pressureGood = pressureMid
    
                pressureTest = pressureGood  # the lowest (converged) pressure that gives valid density
    
                print(("Warning: The initialized density field contains NaN values."
                        "Outlet BC is of transmissive kind, the outlet pressure imposed"
                        "(necessary for velocity initialization) was likely too low. "
                        f"Adjusting the outlet pressure to {pressureTest:.6e} Pa for initialization."))
                fluidState['Pressure'][-1] = pressureTest
                fluidState["Pressure"] = np.interp(
                    meshData['xMeshNodes'],
                    [meshData['xMeshNodes'][0], meshData['xMeshNodes'][-1]],
                    [fluidState['Pressure'][0], fluidState['Pressure'][-1]]
                )
                fluidState["Density"] = fluidModel.computeDensity_p_s(
                    fluidState["Pressure"], entropyField
                    )
            fluidState["Energy"] = fluidModel.computeInternalEnergy_p_s(fluidState["Pressure"], entropyField)

            # Initialize velocity.
            if not isTotalInlet:
                # Static inlet conditions: velocity initialized linearly from 10 to 200 m/s.
                fluidState["Velocity"] = np.interp(
                    meshData['xMeshNodes'], [meshData['xMeshNodes'][0], meshData['xMeshNodes'][-1]], [10, 200]
                )
            else:
                # Deduce which type of total inlet conditions were specified.
                inlet_conditions = config.InletConditionsValues()
                if 0.0 <= inlet_conditions[1] <= 1.0:
                    inletConditionsVars = "pQ"
                else: 
                    inletConditionsVars = "ptTt"
                # Total inlet conditions: u_i = sqrt(2*(h_t - h_s)), with h_s evaluated from pressure + inlet entropy.
                if inletConditionsVars == "ptTt":
                    totalTemperature = config.InletConditionsValues()[1]
                    totalEnthalpy = fluidModel.computeEnthalpy_p_T(
                        fluidState['Pressure'][inletIdx], totalTemperature
                    )
                elif inletConditionsVars == "pQ":
                    totalPressure, totalQuality = config.InletConditionsValues()[:2]
                    totalEnthalpy = fluidModel.computeEnthalpy_p_Q(totalPressure, totalQuality)
    
                totalEnthalpyField = np.full_like(fluidState['Pressure'], totalEnthalpy)
                staticEnthalpyField = fluidModel.computeEnthalpy_p_s(fluidState["Pressure"], entropyField)
                fluidState["Velocity"] = np.sqrt(2 * (totalEnthalpyField - staticEnthalpyField))

            return fluidState

        # use helper functions for initializing the fluid state arrays based 
        # on the expansion device type.
        if config.expansionDeviceType() == "shocktube":
            fluidState = imposeInitialConditionsShocktube(self, config, fluidModel, fluidState)
        elif config.expansionDeviceType() == "nozzle":
            if config.fluidModel() == "real" and \
               ((config.boundaryConditions()[0] == "inlet" and config.boundaryConditions()[1] == "outlet") \
                or (config.boundaryConditions()[0] == "outlet" and config.boundaryConditions()[1] == "inlet") or \
               (config.boundaryConditions()[0] == "inlet" and config.boundaryConditions()[1] == "transparent") or \
                (config.boundaryConditions()[0] == "transparent" and config.boundaryConditions()[1] == "inlet")):
                # linear nozzle initialization is currently only supported for real fluids with inlet, outlet or transparent boundary conditions.
                fluidState = imposeInitialConditionsNozzleLinear(self, meshData, fluidModel, fluidState)
            else:
                # for other cases, initialization is done with constant primitive variable fields, 
                # just like for the shock tube case, hence the same functionality is used. This should
                # be extended in the future.
                fluidState = imposeInitialConditionsShocktube(self, config, fluidModel, fluidState)

        return fluidState

    



    # Initialize the fluid thermodynamic state at the meshNodes. 
            



    def 










    def extractRestartData(self):
        """
        Extract restart data from a previous simulation step. The results/output file acts as a restart file. 
        Information necessary from the last saved simulation step (saving occurs every writeInterval steps):
        - the time elapsed
        - the primitive variable fields
        - a config object containing the full simulation configuration according to the input.ini file used to initialize the simulation (current input.ini can be different, the original values will be used)
        - iteration index

        Arguments
        ---------
        restartFile : str
            The path to the restart file, which contains the state of the system at the last simulation step.
        
        Returns
        -------
        None, but sets the solutionPrimitive attribute of the Driver class to the values stored in the restart file for the last simulation step, 
        and prints a message to the terminal indicating that the initialization from the restart file was successful.
        """
        if self.restartFilePath is None:
            raise ValueError("Restart file path not specified in the configuration file. Please provide a valid restart file path.")
        with open(self.restartFilePath, 'rb') as file:
            restartData = pickle.load(file)

        timeElapsed = restartData['Time']
        solutionPrimitiveRestart = restartData['Primitive']    
        configRestart = restartData['Configuration']
        iterationIndex = restartData['Iteration Counter']

        return timeElapsed, solutionPrimitiveRestart, configRestart, iterationIndex


    
    def prepareOutputPaths(self):
        """
        Prepare the output paths for the results, ensuring that existing files are not overwritten by appending a counter to the filename if needed.

        Arguments
        ---------
        self : Driver
            The Driver instance, which contains the configuration and will store the results path.
        
        Returns
        -------
        None, but sets the resultsPath attribute of the Driver instance to a unique directory for storing results.
        """
        # Prepare results path, if already exists, do not create new directory, which would
        # overwrite the existing one.
        self.workingDir = Path.cwd()
        self.resultsDirectoryName = f"{self.config.ResultsDirectoryName()}_NX_{self.nNodes}"
        self.resultsPath = Path("Results") / self.resultsDirectoryName
        
        if self.restartFilePath is not None:
            # do nothing, this will ensure new iterations 
            # will be appended to the existing results directory
            pass
        elif self.config.OverwriteResults():
            if os.path.exists(self.resultsPath) and os.path.isdir(self.resultsPath):
                shutil.rmtree(self.resultsPath)
        else:
            dum = self.resultsPath
            counter = 1

            while dum.exists():
                dum = self.resultsDirectory / f"{self.resultsDirectoryName}_{counter}"
                counter += 1

            self.resultsPath = dum
        self.resultsPath.mkdir(parents=True, exist_ok=True)
            
        
  
    

    
    

        



    def instantiateConservativeArrays(self):
        """
        Instantiate the dictionaries of np arrays containing evolution in spatial and temporal directions of the conservative variables. The first dimension is space, the second is time.

        Arguments
        ---------
        None

        Returns
        -------
        None, but sets the solutionConservative attribute of the Driver class to a dictionary of 2D np arrays for each conservative variable, (space, time), initialized to zero.
        """
        self.solutionConsNames = ['u1', 'u2', 'u3']
        self.solutionConservative = {}
        for name in self.solutionConsNames:
            self.solutionConservative[name] = np.zeros(self.nNodesHalo)
        


    

        
    
    




    def setBoundaryConditions(self, config, fluidState):
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
        if config.boundaryConditions()[0] == 'reflective':
            self.setReflectiveBoundaryConditions('left')
        elif config.boundaryConditions()[0] == 'transparent':
            self.setTransparentBoundaryConditions('left')
        elif config.boundaryConditions()[0] == 'periodic':
            self.setPeriodicBoundaryConditions('left')
        elif config.boundaryConditions()[0].lower() == 'inlet':
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
        
        # update also the conservative variable arrays based on what has been done on the primitive, this will only update the halo nodes
        # the remainder of the primitive internal field will be the same as the one resulting after updating the solution using the residuals
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
        
        inletConditions = self.config.InletConditionsValues()    
        if self.config.InletConditionsType().lower()=="total":
            # static pressure is the only info taken from the domain 
            pressure = self.solutionPrimitive['Pressure'][iInternal]
            if pressure>=inletConditions[0]: # avoid the problems that can cause
                pressure = 0.9999*inletConditions[0]
            if self.inletConditionsVars == "ptTt":
                density, velocity, energy = self.fluid.computeInletQuantitiesTotal_pt_Tt(pressure, inletConditions[0], inletConditions[1], inletConditions[2])
            elif self.inletConditionsVars == "pQ":
                density, velocity, energy = self.fluid.computeInletQuantitiesTotal_pt_Q(pressure, inletConditions[0], inletConditions[1], inletConditions[2])
        elif self.config.InletConditionsType().lower()=="static":
            if self.fluidModel=='ideal':
                raise ValueError('Static inlet conditions are only supported for the real fluid model')
            pressure = inletConditions[0]
            enthalpy = inletConditions[1]
            # get flow velocity from the domain
            velocity = 2* self.solutionPrimitive['Velocity'][iInternal] - self.solutionPrimitive['Velocity'][iInternal + 1 * (get_sign(iInternal))] # linear extrapolation
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
            pressure = self.config.OutletConditions() # the pressure is the information taken from outside
            velocity = self.solutionPrimitive['Velocity'][iInternal]
            density = self.solutionPrimitive['Density'][iInternal]
            energy = self.fluid.computeInternalEnergy_p_rho(pressure, density)        
            self.solutionPrimitive['Density'][iHalo] = density
            self.solutionPrimitive['Velocity'][iHalo] = velocity
            self.solutionPrimitive['Pressure'][iHalo] = pressure
            self.solutionPrimitive['Energy'][iHalo] = energy
        else:            
            self.setTransparentBoundaryConditions(location) # the boundary is equivalent to a transparent condition
            
            


    def solve(self):
        """
        Solve the equations explicitly in time (forward Euler) using a certain advectionScheme (`Godunov`, `Roe`, `WAF`). high_order
        specifies if applying or not high order reconstruction with limiters. At the moment only type one is working -> simply
        impose high_order=True
        """
        self.entropyFixActive = self.config.EntropyFixActiveBool()
        self.entropyFixCoefficient = self.config.EntropyFixCoefficient()
        advectionScheme = self.config.NumericalScheme()
        isMusclActive = self.config.MUSCLReconstructionBool()
        writeInterval = self.config.WriteInterval()
        printInfoResidualsBool = self.config.PrintInfoResidualsBool()
        
        print()
        print("="*80)
        print(" "*33 + "START SOLVER")
        print("Numerical flux method: %s" %(advectionScheme))
        print("MUSCL reconstruction: %s" %isMusclActive)
        print("Entropy fix active: %s" %self.entropyFixActive)
        if self.config.FluidModel()=='real':
            print("Real Gas model, library: %s" %self.config.FluidLibrary())
        else:
            print("Ideal Gas model")
        if self.entropyFixActive:
            print("Entropy fix coefficient: %s" %self.entropyFixCoefficient)
        print("="*80)
        print()

        # short aliases (shallow copy, will change throughout the iteration loop)
        primitiveOld = copy.deepcopy(self.solutionPrimitive)
        
        # prepare output paths based on config specification
        self.prepareOutputPaths()

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
        self.regroupSingleResults(self.resultsPath)
        print(" "*34 + "END ASSEMBLER")
        print("="*80)
    
    
    def computeResiduals(self, primitives, dt):
        availableLimiters = ['van albada', 'van leer', 'min-mod', 'superbee', 'none']
        
        limiter = self.config.FluxLimiter()
        if limiter not in availableLimiters:
            raise ValueError(f'Limiter not recognized! Available ones are: {availableLimiters}')
        
        advectionScheme = self.config.NumericalScheme()
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
    #     outputDirectoryName = self.config.OutputDirectoryName()
    #     os.makedirs(outputDirectoryName, exist_ok=True)
    #     file_name = self.config.OutputFileName()
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
    

    def generateNozzleAreaTube(self, xTube, filepath):
        nozzleData = np.loadtxt(filepath, skiprows=1, delimiter=',', dtype=float)
        nozzleX = nozzleData[:,0]
        nozzleArea = nozzleData[:,1]
        
        # Linear interpolation with external filling set to area Reference (=Tube area)
        interpolatedNozzleArea = np.interp(xTube, nozzleX, nozzleArea, left=nozzleData[0,1], right=nozzleData[-1,1])
        
        return interpolatedNozzleArea



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
        self.time = np.zeros(nTimes)
        self.solution = {}
        
        print("Regrouping all the results in a single file...")
        for iFile in range(len(files)):
            print(f"Reading File {iFile+1} of {len(files)}")
            with open(filepath / files[iFile], 'rb') as file:
                result = pickle.load(file)
                
                if iFile == 0:
                    nNodesVirtual = result['Primitive']['Pressure'].shape[0]
                    self.xNodesVirtual = result['X Coords']
                    self.areaTube = result['Area Tube']
                    self.iterationCounter = result['Iteration Counter']
                    self.fluid = result['Fluid']
                    self.config = result['Configuration']
                    
                    self.timeVec = np.zeros(nTimes)
                    self.solution['Density'] = np.zeros((nNodesVirtual, nTimes))
                    self.solution['Velocity'] = np.zeros((nNodesVirtual, nTimes))
                    self.solution['Pressure'] = np.zeros((nNodesVirtual, nTimes))
                
                self.timeVec[iFile] = result['Time']
                self.solution['Density'][:, iFile] = result['Primitive']['Density']
                self.solution['Velocity'][:, iFile] = result['Primitive']['Velocity']
                self.solution['Pressure'][:, iFile] = result['Primitive']['Pressure']
        
        globalOutput = {'X Coords': self.xNodesVirtual, 
                        'Area Tube': self.areaTube,
                        'Time': self.timeVec, 
                        'Primitive': self.solution, 
                        'Fluid': self.fluid, 
                        'Configuration': self.config}
        
        print("Replacing all individual files with a single pickle (this could take a while) ...")
        shutil.rmtree(filepath)
        os.makedirs(filepath, exist_ok=True)
        with open(filepath / 'Results.pik', 'wb') as file:
            pickle.dump(globalOutput, file)
        print(f"Regrouped all the times in a single file: {filepath / 'Results.pik'}")4
























