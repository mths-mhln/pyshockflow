import pytest
import numpy as np
from pyshockflow.fluid import FluidReal

@pytest.fixture
def real_fluid():
    return FluidReal(fluid_name='air', fluid_library='CoolProp')
    
def test_computeTemperature_p_rho(real_fluid):
    p = 101325
    rho = 1.225
    T = real_fluid.computeTemperature_p_rho(p, rho)
    assert T == pytest.approx(288.15, abs=0.5)

def test_computeInletQuantities(real_fluid):
    pressure = np.linspace(1E5, 1E7, 10)
    totPressure = pressure*1.5
    totTemperature = 288.15
    direction = 1
    
    for i in range(len(pressure)):
        rho, u, e = real_fluid.computeInletQuantities(pressure[i], totPressure[i], totTemperature, direction)
        s_total = real_fluid.computeEntropy_p_T(totPressure[i], totTemperature)
        s_static = real_fluid.computeEntropy_p_rho(pressure[i], rho)
        assert s_static == pytest.approx(s_total, abs=1e-3)