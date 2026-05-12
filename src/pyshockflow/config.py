import configparser
import os

class Config:
    def __init__(self, config_file='input.ini'):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found.")
        
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(config_file)
    
    def getNumberOfPoints(self):
        return int(self.config_parser.get('SIMULATION', 'NUMBER_POINTS')) 
    
    def getLength(self):
        return float(self.config_parser.get('GEOMETRY', 'LENGTH')) 
    
    def getPressureLeft(self):
        return float(self.config_parser.get('SIMULATION', 'PRESSURE_LEFT')) 
    
    def getPressureRight(self):
        return float(self.config_parser.get('SIMULATION', 'PRESSURE_RIGHT')) 
    
    def getDensityLeft(self):
        return float(self.config_parser.get('SIMULATION', 'DENSITY_LEFT')) 
    
    def getDensityRight(self):
        return float(self.config_parser.get('SIMULATION', 'DENSITY_RIGHT')) 
    
    def getTemperatureLeft(self):
        return float(self.config_parser.get('SIMULATION', 'TEMPERATURE_LEFT')) 
    
    def getTemperatureRight(self):
        return float(self.config_parser.get('SIMULATION', 'TEMPERATURE_RIGHT')) 
    
    def getVelocityLeft(self):
        return float(self.config_parser.get('SIMULATION', 'VELOCITY_LEFT')) 
    
    def getVelocityRight(self):
        return float(self.config_parser.get('SIMULATION', 'VELOCITY_RIGHT')) 
    
    def getCFLMax(self):
        return float(self.config_parser.get('SIMULATION', 'CFL_MAX')) 
    
    def getTimeMax(self):
        return float(self.config_parser.get('SIMULATION', 'TIME_MAX')) 
            
    def getTimeStepMethod(self):
        try:
            return str(self.config_parser.get('SIMULATION', 'TIME_STEP_METHOD')).lower()
        except:
            return 'constant'
    
    def getFluidName(self):
        return str(self.config_parser.get('FLUID', 'FLUID_NAME'))
    
    def getFluidModel(self):
        return str(self.config_parser.get('FLUID', 'FLUID_MODEL')).lower() 
    
    def getFluidGamma(self):
        return float(self.config_parser.get('FLUID', 'FLUID_GAMMA'))
    
    def getGasRConstant(self):
        return float(self.config_parser.get('FLUID', 'GAS_R_CONSTANT'))
    
    def getInterfaceLocation(self):
        return float(self.config_parser.get('GEOMETRY', 'INTERFACE_LOCATION'))
    
    def getBoundaryConditions(self):
        left = str(self.config_parser.get('SIMULATION', 'BOUNDARY_CONDITION_LEFT')).lower() 
        right = str(self.config_parser.get('SIMULATION', 'BOUNDARY_CONDITION_RIGHT')).lower() 
        return (left, right)

    def getInletConditions(self):
        res = str(self.config_parser.get('SIMULATION', 'INLET_CONDITIONS')).lower() 
        res = [float(value.strip()) for value in res.split(',')]
        if self.getInletConditionsType().lower()=='total':
            if len(res)!=3:
                raise ValueError("For total inlet conditions, INLET_CONDITIONS must have 3 values: total pressure, total temperature and direction.")
        else:
            if len(res)!=2:
                raise ValueError("For static inlet conditions, INLET_CONDITIONS must have 2 values: static pressure and enthalpy.")
        return res

    def getInletConditionsType(self):
        try:
            res = str(self.config_parser.get('SIMULATION', 'INLET_CONDITIONS_TYPE')).lower() 
            if res in ['total', 'static']:
                return res
            else:
                raise ValueError("INLET_CONDITIONS_TYPE must be either 'total' or 'static'.")
        except:
            return 'total' # total by default
    
    def getOutletConditions(self):
        res = float(self.config_parser.get('SIMULATION', 'OUTLET_CONDITIONS'))
        return res
    
    def getNumericalScheme(self):
        return str(self.config_parser.get('SIMULATION', 'NUMERICAL_SCHEME')).lower() 
    
    def getFluxLimiter(self):
        try:
            return str(self.config_parser.get('SIMULATION', 'FLUX_LIMITER')).lower() 
        except:
            return 'van albada'
    
    def getResultsDirectoryName(self):
        return str(self.config_parser.get('OUTPUT', 'RESULTS_DIRECTORY_NAME')) 
    
    def showAnimation(self):
        res = str(self.config_parser.get('OUTPUT', 'SHOW_ANIMATION')).lower() 
        if res=='yes' or res=='true':
            return True
        else:
            return False
    
    def isMusclActive(self):
        try:
            res = str(self.config_parser.get('SIMULATION', 'MUSCL_RECONSTRUCTION')).lower() 
            if res=='yes' or res=='true':
                return True
            else:
                return False
        except:
            return False # false by default
    
    
    def isMeshRefined(self):
        try:
            res = str(self.config_parser.get('SIMULATION', 'MESH_REFINEMENT')).lower() 
            if res=='yes' or res=='true':
                return True
            else:
                return False
        except:
            return False # false by default
    
    def isEntropyFixActive(self):
        try:
            res = str(self.config_parser.get('SIMULATION', 'ENTROPY_FIX_ACTIVE')).lower() 
            if res=='yes' or res=='true':
                return True
            else:
                return False
        except:
            return True # true by default
    
    def getEntropyFixCoefficient(self):
        try:
            res = float(self.config_parser.get('SIMULATION', 'ENTROPY_FIX_COEFFICIENT'))
            return res
        except:
            return 0.2 # 0.2 by default
    
    def getFluidLibrary(self):
        return str(self.config_parser.get('FLUID', 'FLUID_LIBRARY'))
    
    def adaptMeshRefinementExtremities(self):
        try:
            res = str(self.config_parser.get('SIMULATION', 'ADAPT_MESH_REFINEMENT')).lower() 
            if res=='yes' or res=='true':
                return True
            else:
                return False
        except:
            return False #  by default
    
    def getRefinementBoundaries(self):
        start = float(self.config_parser.get('SIMULATION', 'X_START_REFINEMENT')) 
        end = float(self.config_parser.get('SIMULATION', 'X_END_REFINEMENT')) 
        return (start, end)
    
    def getNumberPointsRefinement(self):
        return int(self.config_parser.get('SIMULATION', 'NUMBER_POINTS_REFINEMENT')) 
        
    
    def getTopology(self):
        try:
            return str(self.config_parser.get('GEOMETRY', 'TOPOLOGY')).lower()
        except:
            return 'default' 
    
    def getNozzleFilePath(self):
        return str(self.config_parser.get('GEOMETRY', 'NOZZLE_FILEPATH'))
    
    
    def getAreaReference(self):
        try:
            return float(self.config_parser.get('GEOMETRY', 'REFERENCE_AREA')) 
        except:
            return 1.0 # default
    
    
    def getWriteInterval(self):
        try:
            return int(self.config_parser.get('OUTPUT', 'WRITE_INTERVAL')) 
        except:
            return 250 # default value

    def getOverwriteResults(self):
        try:
            res = str(self.config_parser.get('OUTPUT', 'OVERWRITE_RESULTS')).lower() 
            if res=='yes' or res=='true':
                return True
            else:
                return False
        except:
            return False #  by default
        
    def getPrintInfoResidualsBool(self):
        try:
            res = str(self.config_parser.get('OUTPUT', 'PRINT_INFO_RESIDUALS')).lower() 
            if res=='yes' or res=='true':
                return True
            else:
                return False
        except:
            return False #  by default