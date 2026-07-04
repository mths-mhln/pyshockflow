import pytest

from unittest.mock import MagicMock


@pytest.fixture
def mock_config():
    """
    Mock of Config based on a nozzle simulation with CO2/RefProp real gas,
    inlet/transparent boundary conditions, and Roe-Arabi scheme.
    """
    cfg = MagicMock()
 
    # [GEOMETRY]
    cfg.getLength.return_value = 0.09837
    cfg.getInterfaceLocation.return_value = 0.05
    cfg.getTopology.return_value = "nozzle"
    cfg.getNozzleFilePath.return_value = "inputs/nozzle_geometries/nozzle_lettieri.csv"
    cfg.getAreaReference.return_value = 1.0
 
    # [SIMULATION] — initial states
    cfg.getNumberOfPoints.return_value = 200
    cfg.getTimeMax.return_value = 0.0500
    cfg.getTemperatureLeft.return_value = 243.3024
    cfg.getTemperatureRight.return_value = 243.3024
    cfg.getVelocityLeft.return_value = 100.0
    cfg.getVelocityRight.return_value = 100.0
    cfg.getPressureLeft.return_value = 30000.0
    cfg.getPressureRight.return_value = 30000.0
 
    # [SIMULATION] — numerics
    cfg.getNumericalScheme.return_value = "roe_arabi"
    cfg.getCFLMax.return_value = 0.7
    cfg.getTimeStepMethod.return_value = "constant"
    cfg.getFluxLimiter.return_value = "van albada"
 
    # [SIMULATION] — boundary conditions
    cfg.getBoundaryConditions.return_value = ("inlet", "transparent")
    cfg.getInletConditions.return_value = [5896000.0, 314.7, 1.0]
    cfg.getInletConditionsType.return_value = "total"
 
    # [SIMULATION] — optional reconstructions / fixes
    cfg.isMusclActive.return_value = False
    cfg.isMeshRefined.return_value = False
    cfg.isEntropyFixActive.return_value = True
    cfg.getEntropyFixCoefficient.return_value = 0.2
    cfg.adaptMeshRefinementExtremities.return_value = False
 
    # [FLUID]
    cfg.getFluidName.return_value = "CO2"
    cfg.getFluidModel.return_value = "real"
    cfg.getFluidLibrary.return_value = "RefProp"
    cfg.getPropertyExtractionMethod.return_value = "fluid"
    cfg.getFluidGamma.return_value = 1.4          # commented out in ini but kept as sane default
    cfg.getGasRConstant.return_value = 287.05     # same
 
    # [OUTPUT]
    cfg.getResultsDirectoryName.return_value = "output_RefProp"
    cfg.showAnimation.return_value = False
    cfg.getWriteInterval.return_value = 100
    cfg.getPrintInfoResidualsBool.return_value = True
    cfg.getOverwriteResults.return_value = True
 
    return cfg


@pytest.fixture
def mock_fluid():
    """Return a MagicMock that satisfies every fluid.compute* call Driver makes."""
    fluid = MagicMock()
    fluid.computeTemperature_p_rho.return_value = 300.0
    fluid.computeDensity_p_T.return_value = 1.2
    fluid.computeStaticEnergy_p_rho.return_value = 2.15e5
    return fluid