import pytest
from pyshockflow.fluid import FluidIdeal

@pytest.fixture
def ideal_fluid():
    return FluidIdeal(gmma=1.0, Rgas=1.0)

def test_computeTemperature_p_rho(ideal_fluid):
    p = 100
    rho = 1
    T = ideal_fluid.computeTemperature_p_rho(p, rho)
    assert T == pytest.approx(100, abs=1e-6)
    