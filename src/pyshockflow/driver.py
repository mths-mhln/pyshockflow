import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import csv
import sys
import copy
import timeit
import shutil
from pathlib import Path
from pyshockflow import RiemannProblem
from pyshockflow import AdvectionRoeBase, AdvectionRoeArabi, AdvectionRoeVinokur
from pyshockflow import FluidIdeal, FluidReal
from pyshockflow.output import Output
from pyshockflow.math_utils import getPrimitivesFromConservatives, getConservativesFromPrimitives, computeAdvectionFluxFromConservatives


class Driver:
    def __init__(self, config = None, restartFilePath=None):
        """
        - Extract geometry, simulation and fluid information from teh configuration file. 
        - Compute missing thermodynamic properties based on those provided.
        - Generate physical (=internal) and virtual geometry (with halo nodes for BCs and possible area variation for quasi 1D simulation).
        - Prepare output paths, print information to the terminal that will be captured and stored in the log file by the subprocess.
        - Instantiate arrays to store primitive and conservation variables. 
        - Set field parameters according to those at the last simulation step if a restart file is provided, 
          otherwise impose initial conditions based on the left and right values provided in the configuration file.
        - Impose boundary conditions to the halo nodes based on the type specified in the configuration file.

        Arguments
        ---------
        config : Config
            The configuration class, containing methods for extracting information from the configuration file.

        Returns
        -------
        - None, stores all the relevant information as attributes of the Driver class.
        """
        self.restartFilePath = restartFilePath
        if self.restartFilePath is not None:
            timeElapsed, solutionPrimitiveRestart, configRestart, iterationIndex = self.extractRestartData()
            print(f"Restarting simulation from file {self.restartFilePath} at iteration {iterationIndex} and time elapsed {timeElapsed:.6e} s")
            if config is None:
                self.config = configRestart
            else:
                self.config = config
        else:    
            self.config = config
        self.topology = self.config.getTopology()
        self.fluidName = self.config.getFluidName()
        self.fluidModel = self.config.getFluidModel()
        
        if self.fluidModel.lower()=='ideal':
            self.gmma = self.config.getFluidGamma()
            self.Rgas = self.config.getGasRConstant()
            self.fluid = FluidIdeal(self.gmma,self.Rgas)
        elif self.fluidModel.lower()=='real':
            fluidLibrary = self.config.getFluidLibrary()
            availFluidLibs = ['StanMix', 'GasMix', 'PCP-SAFT', 'RefProp', 'qPCP-SAFT', 'HOGC-PCP-SAFT',
                                'CoolProp', 'REFPROP', 'HEOS',
                                'Humid Air', 'Humid Air Mix', 'LuT', 'feos::HOGC-PCP-SAFT']
            if fluidLibrary not in availFluidLibs:
                raise ValueError(f"Invalid fluid library: {fluidLibrary}. Must be one of {availFluidLibs}")
            self.fluid = FluidReal(self.fluidName, fluidLibrary, self.config.getPropertyExtractionMethod(), False)
        
        # fluid initial states
        self.pressureLeft = self.config.getPressureLeft()
        self.pressureRight = self.config.getPressureRight()
        
        try:
            self.densityLeft = self.config.getDensityLeft()
            self.densityRight = self.config.getDensityRight()
            self.temperatureLeft = self.fluid.computeTemperature_p_rho(self.pressureLeft, self.densityLeft)
            self.temperatureRight = self.fluid.computeTemperature_p_rho(self.pressureRight, self.densityRight)
        except:
            self.temperatureLeft = self.config.getTemperatureLeft()
            self.temperatureRight = self.config.getTemperatureRight()
            self.densityLeft = self.fluid.computeDensity_p_T(self.pressureLeft, self.temperatureLeft)
            self.densityRight = self.fluid.computeDensity_p_T(self.pressureRight, self.temperatureRight)
        
        self.velocityLeft = self.config.getVelocityLeft()
        self.velocityRight = self.config.getVelocityRight()
        self.energyLeft = self.fluid.computeStaticEnergy_p_rho(self.pressureLeft, self.densityLeft)
        self.energyRight = self.fluid.computeStaticEnergy_p_rho(self.pressureRight, self.densityRight)
        
        # geometry
        self.length = self.config.getLength()
        self.nNodes = self.config.getNumberOfPoints()
        xNodes = self.generatePhysicalGeometry(self.length, self.nNodes)
        self.generateVirtualGeometry(xNodes)

        # Prepare results path
        self.resultsDirectory = Path("Results")
        self.resultsDirectory.mkdir(parents=True, exist_ok=True)
        self.workingDir = Path.cwd()
        self.resultsDirectoryName = f"{self.config.getResultsDirectoryName()}_NX_{self.nNodes}"
        self.resultsPath = self.resultsDirectory / self.resultsDirectoryName
        
        # Time related information
        self.cflMax = self.config.getCFLMax()
        self.timeMax = self.config.getTimeMax()
        
        # Boundary Conditions
        self.boundaryType = self.config.getBoundaryConditions()    
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
            self.solutionPrimitive = solutionPrimitiveRestart
            self.time = timeElapsed
            self.iterationIndex = iterationIndex
        else:
            self.imposeInitialConditions()
        self.setBoundaryConditions()
        if self.restartFilePath is not None:
            self.solve()
    
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
        if self.restartFilePath is not None:
            # do nothing, append new iterations to the current working directory
            pass
        elif self.config.getOverwriteResults():
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
            
        
                
    def generatePhysicalGeometry(self, length, nodes):
        """
        Generate the physical geometry of the problem, which consists of the coordinates of the nodes along the x-axis. 
        The distribution of the nodes can be uniform or non-uniform based on the mesh refinement settings in the configuration file.
        Mesh stretching adds a single node close to the interface between regular and refined. 

        Arguments
        ---------
        length : float
            The length of the domain along the 1D (x) axis.
        nodes : int
            The total number of nodes in the physical geometry (excluding halo nodes).

        Returns
        -------
        xNodes : numpy array
            The coordinates of the nodes along the 1D (x) axis, including any mesh refinement if specified in the configuration.
        """
        isMeshRefined = self.config.isMeshRefined()
        if isMeshRefined is False:
            xNodes = np.linspace(0, length, nodes)
        else:
            refinementCoords = self.config.getRefinementBoundaries()
            print("Mesh is refined between the two boundaries [m]: ", refinementCoords)
            
            pointsRefinement = self.config.getNumberPointsRefinement()
            totalPoints = nodes
            pointsOutside = totalPoints-pointsRefinement
            lengthUpstream = refinementCoords[0]
            lengthDownstream = length-refinementCoords[1]
            try:
                pointsUpstream = int(pointsOutside*(lengthUpstream)/(lengthUpstream+lengthDownstream))
            except ZeroDivisionError:
                # refinement starts at the beginning of the domain and/or ends at the end of the domain, so all points outside are either upstream or downstream
                pointsUpstream = 0
            pointsDownstream = pointsOutside-pointsUpstream
            
            # case in which the refinement is internal
            if pointsDownstream>0 and pointsUpstream>0:
                xUpstream = np.linspace(0, refinementCoords[0], pointsUpstream+1)
                xRefinement = np.linspace(refinementCoords[0], refinementCoords[1], pointsRefinement+1)
                xDownstream = np.linspace(refinementCoords[1], length, pointsDownstream)
                if self.config.adaptMeshRefinementExtremities():
                    xUpstream = self.computeStretchedGridPoints(xUpstream, xRefinement, 'upstream')
                    xDownstream = self.computeStretchedGridPoints(xDownstream, xRefinement, 'downstream')
                xNodes = np.concatenate((xUpstream[0:-1], xRefinement[0:-1], xDownstream))
                
            elif pointsUpstream>0 and pointsDownstream==0: # the refinement finish with the end of the domain
                xUpstream = np.linspace(0, refinementCoords[0], pointsUpstream+1)
                xRefinement = np.linspace(refinementCoords[0], refinementCoords[1], pointsRefinement+1)
                if self.config.adaptMeshRefinementExtremities():
                    xUpstream = self.computeStretchedGridPoints(xUpstream, xRefinement, 'upstream')  
                xNodes = np.concatenate((xUpstream[0:-1], xRefinement))
            
            elif pointsUpstream==0 and pointsDownstream>0: # the refinement starts with the domain
                xRefinement = np.linspace(refinementCoords[0], refinementCoords[1], pointsRefinement+1)
                xDownstream = np.linspace(refinementCoords[1], length, pointsDownstream)
                if self.config.adaptMeshRefinementExtremities():
                    xDownstream = self.computeStretchedGridPoints(xDownstream, xRefinement, 'downstream')
                xNodes = np.concatenate((xRefinement[0:-1], xDownstream))
            
            else:
                if pointsOutside == 0:
                    raise ValueError('The number of points in the refinement is equal to the total number of points, so no points are left for the regular mesh. Please reduce the number of points in the refinement.')
                raise ValueError('The refinement is ill-positioned. Please locate it internally to the domain, or at one of the extremities')
            
            self.nNodes = len(xNodes)
        
        return xNodes  
    
    

    def computeGridSpacing(self, xNodes):
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
    
    

    def computeStretchedGridPoints(self, xCoords, xRefinement, location):    
        """
        Compute the stretched grid points to add close to the interface between regular and refined mesh, if the option is activated in the configuration file.
        The stretching is done by adding a single node at the middle of the distance between the last regular node and the first refined node, and iterating 
        until the minimum spacing in the regular mesh is smaller than the minimum spacing in the refined mesh.

        Arguments
        ---------
        xCoords : numpy array
            The coordinates of the nodes along the 1D (x) axis for the regular mesh, excluding halo nodes.
        xRefinement : numpy array
            The coordinates of the nodes along the 1D (x) axis for the refined mesh, excluding halo nodes.
        location : str
            The location of the refinement, either 'upstream' or 'downstream'.

        Returns
        -------
        xNew : numpy array
            The coordinates of the nodes along the 1D (x) axis for the regular mesh, including the stretched nodes, excluding halo nodes.
        """  
        if location=='upstream':
            xNew = xCoords[0:-1].copy()
            dxMin = np.min(self.computeGridSpacing(xNew))
            dxRef = np.min(self.computeGridSpacing(xRefinement))
            while (dxMin-dxRef>0):
                newPoint =  xNew[-1] + (xRefinement[0]-xNew[-1])*0.5
                xNew = np.append(xNew, newPoint)
                dxMin = np.min(self.computeGridSpacing(xNew))
            xNew[-1] = xRefinement[0]
            # xNew = np.append(xNew, xRefinement[0])
        else:
            xNew = xCoords[1:].copy()
            dxMin = np.min(self.computeGridSpacing(xNew))
            dxRef = np.min(self.computeGridSpacing(xRefinement))
            while (dxMin-dxRef>0):
                newPoint =  xNew[0] - (+xNew[0] - xRefinement[-1])*0.5
                xNew = np.insert(xNew, 0, newPoint)
                dxMin = np.min(self.computeGridSpacing(xNew))
            xNew[0] = xRefinement[-1]
            # xNew = np.insert(xNew, 0, xRefinement[-1])
        return xNew
        
        
    
    def generateVirtualGeometry(self, xNodes):
        """
        Generate the virtual geometry of the problem. Virtual geometry consists of physical geometry (the nodes along the x-axis) and halo nodes (the nodes added for boundary conditions).
        Virtual geometry also includes possible area variation for quasi 1D simulation.

        Parameters
        ----------
        xNodes : numpy array
            The coordinates of the nodes along the 1D (x) axis, excluding halo nodes.

        Returns
        -------
        None, but sets the xNodesVirtual attribute of the Driver class to the coordinates of the nodes along the 1D (x) axis, including halo nodes, 
        and the areaTube attribute to the area variation along the x-axis if specified in the configuration file.
        """
        self.xNodes = xNodes
        self.nNodesHalo = self.nNodes+2
        self.xNodesVirtual = np.zeros(self.nNodesHalo)
        self.xNodesVirtual[1:-1] = self.xNodes
        self.xNodesVirtual[0] = self.xNodes[0] - (xNodes[1]-xNodes[0])
        self.xNodesVirtual[-1] = self.xNodes[-1] + (xNodes[-1]-xNodes[-2])
        self.areaReference = self.config.getAreaReference()
        self.dx = self.computeGridSpacing(self.xNodesVirtual)
        
        if self.topology.lower()=='default':
            print("The simulation proceeds with default topology: constant area")
            self.areaTube = np.zeros_like(self.xNodesVirtual)+self.areaReference
        elif self.topology.lower()=='nozzle':
            print(f"The simulation topology is: nozzle. Reading the coordinates from the nozzle file {self.config.getNozzleFilePath()}")
            self.areaTube = self.readNozzleFile(self.xNodesVirtual, self.config.getNozzleFilePath())
        else:
            raise ValueError('Unknown topology type')
        
        self.dAreaTube_dx = np.gradient(self.areaTube, self.xNodesVirtual)
        
        

    def instantiatePrimitiveArrays(self):
        """
        Instantiate the dictionaries of np arrays containing evolution in spatial and temporal directions of the primitive variables. 
        The first dimension is space, the second is time. The primitive variables are Density, Velocity, Pressure and Energy.

        Arguments
        ---------
        None

        Returns
        -------
        None, but sets the solutionPrimitive attribute of the Driver class to a dictionary of 2D np arrays for each primitive variable, (space, time), initialized to zero.
        """
        self.solutionNames = ['Density', 'Velocity', 'Pressure', 'Energy']
        self.solutionPrimitive = {}
        for name in self.solutionNames:
            self.solutionPrimitive[name] = np.zeros(self.nNodesHalo)



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
        

    def imposeInitialConditions(self):
        """
        Method reserved for shocktube experiments. Intended to initialize the state of the system at either side of the interface. All information is contained in the configuration file. 
        The interface is imposed into the array through the copyInitialState method.

        Arguments
        ---------
        None

        Returns
        -------
        None, but sets the solutionPrimitive attribute of the Driver class to the initial conditions based on the left and right values provided in the configuration file, 
        and prints these values to the terminal.
        """
        initialConditions = {'Density': np.array([self.densityLeft, self.densityRight]), 
                             'Velocity': np.array([self.velocityLeft, self.velocityRight]), 
                             'Pressure': np.array([self.pressureLeft, self.pressureRight]),
                             'Energy': np.array([self.energyLeft, self.energyRight])}
        
        for name in self.solutionNames:
            self.solutionPrimitive[name] = self.copyInitialState(initialConditions[name][0], initialConditions[name][1])
        
        print(f"Initial L/R density values [kg/m3]:          ({self.densityLeft:.6e}, {self.densityRight:.6e})")
        print(f"Initial L/R velocity values [m/s]:           ({self.velocityLeft:.6e}, {self.velocityRight:.6e})")
        print(f"Initial L/R pressure values [Pa]:            ({self.pressureLeft:.6e}, {self.pressureRight:.6e})")
        print(f"Initial L/R temperature values [K]:          ({self.temperatureLeft:.6e}, {self.temperatureRight:.6e})")
        print(f"Initial L/R energy values [J/kg]:            ({self.energyLeft:.6e}, {self.energyRight:.6e})")
        


    # def initialConditionsArrays(self, dictIn):
    #     """
    #     Method currently not used anywhere in Driver logic.
        
    #     Arguments
    #     ---------
    #     dictIn : dict
    #         A dictionary containing the initial values of the primitive variables, with keys corresponding to the variable names and values corresponding to the initial values.
        
    #     Returns
    #     -------
    #     None, but sets the solutionPrimitive attribute of the Driver class to the initial conditions based on the values provided in the dictIn dictionary,
    #     """
    #     dictIn['Energy'] = dictIn['Pressure'] / (self.gmma - 1) / dictIn['Density']
    #     for name in self.solutionNames:
    #         self.solutionPrimitive[name][:, 1:-1] = dictIn[name]
    
    

    # def plotGridGeometry(self, trueAspectRatio=True, pointsToJump=1, save_filename=None):
    #     """
    #     Method currently not used anywhere in Driver logic. 
    #     Arguments
    #     ---------
    #     trueAspectRatio : bool, optional
    #         Whether to set the aspect ratio of the plot to be equal, by default True.
    #     pointsToJump : int, optional
    #         The number of points to jump when plotting the vertical lines representing the grid, by default 1 (plot all points).
    #     save_filename : str, optional
    #         The filename to save the plot as a pdf file, by default None (do not save the plot).

    #     Returns
    #     -------
    #     None, but plots the grid geometry and saves it as a pdf file if a filename is provided.
    #     """
    #     diameter = np.sqrt(4*self.areaTube/np.pi)
    #     yLower = np.zeros_like(self.xNodesVirtual)-diameter/2
    #     yUpper = diameter/2
        
    #     plt.figure()
    #     plt.plot(self.xNodesVirtual, yLower, 'k')
    #     plt.plot(self.xNodesVirtual, yUpper, 'k')
    #     nPointsPic = 10
    #     for i in range(0, len(diameter), pointsToJump):
    #         plt.plot(np.zeros(nPointsPic)+self.xNodesVirtual[i], np.linspace(yLower[i], yUpper[i], nPointsPic), '-k', lw=0.5)
    #     plt.xlabel(r'$x \ \rm{[m]}$')
    #     plt.ylabel(r'$r \ \rm{[m]}$')
        
    #     if trueAspectRatio:
    #         ax = plt.gca()
    #         ax.set_aspect('equal')
        
    #     if save_filename is not None:
    #         plt.savefig(save_filename + '.pdf', bbox_inches='tight')



    def copyInitialState(self, fL, fR):
        """
        Method reserved for shocktube experiments. Intended to initialize the state of the system at either side of the interface. All information is contained in the configuration file.
        Both physical and virtual nodes are initialized. Later the virtual nodes will be overwritten to comply with the boundary conditions. 

        Arguments
        ---------
        fL : float
            The value of the primitive variables on the left side of the interface.
        fR : float
            The value of the primitive variables on the right side of the interface.

        Returns
        -------
        f : numpy.ndarray
            An array containing the initialized primitive variables for all nodes in the computational domain (physical + virtual).
        """
        xInterface = self.config.getInterfaceLocation()
        assert(xInterface>0), f"The interface must be located within 0 and {self.length}"
        assert(xInterface<self.length), f"The interface must be located within 0 and {self.length}"
        
        f = np.zeros_like(self.xNodesVirtual)
        for i in range(len(self.xNodesVirtual)):
            if self.xNodesVirtual[i] <= xInterface:
                f[i] = fL
            else:
                f[i] = fR
        return f



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
        
        inletConditions = self.config.getInletConditions()    
        if self.config.getInletConditionsType().lower()=="total":
            totalPressure = inletConditions[0]
            totalTemperature = inletConditions[1]
            direction = inletConditions[2]
            # static pressure is the only info taken from the domain 
            pressure = self.solutionPrimitive['Pressure'][iInternal]
            if pressure>=totalPressure: # avoid the problems that can cause
                pressure = 0.99*totalPressure 
            density, velocity, energy = self.fluid.computeInletQuantitiesTotal(pressure, totalPressure, totalTemperature, direction)
        elif self.config.getInletConditionsType().lower()=="static":
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
            pressure = self.config.getOutletConditions() # the pressure is the information taken from outside
            velocity = self.solutionPrimitive['Velocity'][iInternal]
            density = self.solutionPrimitive['Density'][iInternal]
            energy = self.fluid.computeStaticEnergy_p_rho(pressure, density)        
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
        self.entropyFixActive = self.config.isEntropyFixActive()
        self.entropyFixCoefficient = self.config.getEntropyFixCoefficient()
        advectionScheme = self.config.getNumericalScheme()
        isMusclActive = self.config.isMusclActive()
        writeInterval = self.config.getWriteInterval()
        printInfoResidualsBool = self.config.getPrintInfoResidualsBool()
        
        print()
        print("="*80)
        print(" "*33 + "START SOLVER")
        print("Numerical flux method: %s" %(advectionScheme))
        print("MUSCL reconstruction: %s" %isMusclActive)
        print("Entropy fix active: %s" %self.entropyFixActive)
        if self.config.getFluidModel()=='real':
            print("Real Gas model, library: %s" %self.config.getFluidLibrary())
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
        
        limiter = self.config.getFluxLimiter()
        if limiter not in availableLimiters:
            raise ValueError(f'Limiter not recognized! Available ones are: {availableLimiters}')
        
        advectionScheme = self.config.getNumericalScheme()
        MUSCL = self.config.isMusclActive()
        
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
                         'Fluid': self.fluid,
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
    























