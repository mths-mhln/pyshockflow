import sys
import pickle
import pytest

import pandas as pd
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from scipy.optimize import newton

from pyshockflow.fluid import FluidReal
import fluid_properties.fluid_properties as FP

@pytest.fixture
def fluid_real_obj(request):
    fluid_name = request.param
    return FluidReal(fluid_name, "REFPROP", "abstractstate_v2")

@pytest.mark.parametrize(
    ("fluid_real_obj", "verification_data_path"),
    (
        ("CO2", "data/test_fluid/data_test_fluid_CO2.pkl"),
        ("R1234ze(E)", "data/test_fluid/data_test_fluid_R1234ze(E).pkl"),
    ),
    # Don't pass this value straight into the test function. Instead, find the fixture
    # called fluid_real_obj, and feed this value into that fixture as request.param
    indirect=["fluid_real_obj"]
)
def test_computeSoundSpeed_p_rho(fluid_real_obj, verification_data_path):
    """ 
    Pytest to verify the computeSoundSpeed_p_rho method of the FluidReal class.
    For two fluids of interest, five test cases will be run: 
    1) an isentropic expansion from liquid to liquid-vapor mixture,
    2) an isothermal process from liquid-vapor mixture to vapor,
    3) a set of thdy states running parallel to the saturation dome at +001% it's temperature value
    4) a set of thdy states running parallel to the saturation dome at -001% it's temperature value
    5) a set of thdy states running parallel to the saturation dome at +000% it's temperature value
    The latter three check if the function behaves well close to the saturation dome and 
    returns single phase speeds of sound or two-phase speeds of sound as appropriate.
    """
    with open(verification_data_path, "rb") as f:
        data = pickle.load(f)

    print(f"fluid: {fluid_real_obj.fluid}, data: {data}")
    
    # data structure
    #{
    # "isentropic_expansion": {"thdy_coords": np.2darray([...]), "sound_speed": np.1darray([...])} 
    # "isothermal_process": {"thdy_coords": np.2darray([...]), "sound_speed": np.1darray([...])}
    # "parallel_plus_001": {"thdy_coords": np.2darray([...]), "sound_speed": np.1darray([...])}
    # "parallel_minus_001": {"thdy_coords": np.2darray([...]), "sound_speed": np.1darray([...])}
    # "parallel_000": {"thdy_coords": np.2darray([...]), "sound_speed": np.1darray([...])}
    # }
    # where the thdy_coords are S, T coordinates.

    # assert case 1 - isentropic expansion
    thdy_coords = data["isentropic_expansion"]["thdy_coords"]
    p = FP.PropsSI("P", "T", thdy_coords[:, 1], "S", thdy_coords[:, 0], fluid_real_obj.fluid)
    rho = FP.PropsSI("D", "T", thdy_coords[:, 1], "S", thdy_coords[:, 0], fluid_real_obj.fluid)
    expected_sound_speed = data["isentropic_expansion"]["sound_speed"]
    sound_speed = fluid_real_obj.computeSoundSpeed_p_rho(p, rho)
    assert sound_speed == pytest.approx(expected_sound_speed, rel=1e-3)

    # assert case 2 - isothermal process
    thdy_coords = data["isothermal_process"]["thdy_coords"]
    p = FP.PropsSI("P", "T", thdy_coords[:, 1], "S", thdy_coords[:, 0], fluid_real_obj.fluid)
    rho = FP.PropsSI("D", "T", thdy_coords[:, 1], "S", thdy_coords[:, 0], fluid_real_obj.fluid)
    expected_sound_speed = data["isothermal_process"]["sound_speed"]
    sound_speed = fluid_real_obj.computeSoundSpeed_p_rho(p, rho)
    assert sound_speed == pytest.approx(expected_sound_speed, rel=1e-3)

    # assert case 3 - parallel_plus_001
    thdy_coords = data["parallel_plus_001"]["thdy_coords"]
    p = FP.PropsSI("P", "T", thdy_coords[:, 1], "S", thdy_coords[:, 0], fluid_real_obj.fluid)
    rho = FP.PropsSI("D", "T", thdy_coords[:, 1], "S", thdy_coords[:, 0], fluid_real_obj.fluid)
    expected_sound_speed = data["parallel_plus_001"]["sound_speed"]
    sound_speed = fluid_real_obj.computeSoundSpeed_p_rho(p, rho)
    assert sound_speed == pytest.approx(expected_sound_speed, rel=1e-3)

    # assert case 4 - parallel_minus_001
    thdy_coords = data["parallel_minus_001"]["thdy_coords"]
    p = FP.PropsSI("P", "T", thdy_coords[:, 1], "S", thdy_coords[:, 0], fluid_real_obj.fluid)
    rho = FP.PropsSI("D", "T", thdy_coords[:, 1], "S", thdy_coords[:, 0], fluid_real_obj.fluid)
    expected_sound_speed = data["parallel_minus_001"]["sound_speed"]
    sound_speed = fluid_real_obj.computeSoundSpeed_p_rho(p, rho)
    assert sound_speed == pytest.approx(expected_sound_speed, rel=1e-3)

    # assert case 5 - parallel_000
    thdy_coords = data["parallel_000"]["thdy_coords"]
    p = FP.PropsSI("P", "T", thdy_coords[:, 1], "S", thdy_coords[:, 0], fluid_real_obj.fluid)
    rho = FP.PropsSI("D", "T", thdy_coords[:, 1], "S", thdy_coords[:, 0], fluid_real_obj.fluid)
    print("pressures: ", p)
    print("densities: ", rho)
    expected_sound_speed = data["parallel_000"]["sound_speed"]
    sound_speed = fluid_real_obj.computeSoundSpeed_p_rho(p, rho)
    assert sound_speed == pytest.approx(expected_sound_speed, rel=1e-3)
    


@pytest.mark.parametrize(
    ("fluid_real_obj", "verification_data_path"),
    (("Water", "data/test_fluid/HEM_sos_benchmark_of_dem_fig_1b.pkl"),),
    indirect=["fluid_real_obj"],
)
def test_computeSoundSpeed_p_rho_de_lorenzo(fluid_real_obj, verification_data_path):
    """
    Verification of function against figure 1b of Marco De Lorenzo "Benchmark of the Delayed Equilibrium Model". 
    Pertains variation of HEM sound speed with volume fraction of vapour for pressure 0.1 MPa and water as working fluid. 
    """
    with open(verification_data_path, "rb") as f:
        data = pickle.load(f)
    
    # data structure
    # {"alpha_V": np.1darray([...]), "sound_speed": np.1darray([...])}

    # Test requires some additional calculations to convert quality into void fraction
    alpha_V = data["alpha_V"]
    # the next parameters are known from how the test case was constructed.
    p = np.ones_like(alpha_V) * 0.1e6  # 0.1 MPa
    rho_v = FP.PropsSI("D", "P", p, "Q", np.ones_like(alpha_V), fluid_real_obj.fluid)

    # solve equation iteratively for rho
    def func(Q, alpha_v, rho_v, p, fluid_real_obj):
        if np.any(Q < 0) or np.any(Q > 1):
            return np.inf  # Return a large number to indicate an invalid solution
        else:
            return (alpha_v / Q) * rho_v - FP.PropsSI("D", "P", p, "Q", Q, fluid_real_obj.fluid)
    
    Q = newton(func, x0=np.full_like(alpha_V, 1e-6),
            args=(alpha_V, rho_v, p, fluid_real_obj))

    rho = FP.PropsSI("D", "P", p, "Q", Q, fluid_real_obj.fluid)
    
    # compute the sound speed using the FluidReal method
    soundSpeed_HEM = fluid_real_obj.computeSoundSpeed_p_rho(p, rho)
    
    # compare the computed sound speed with the expected sound speed from the verification data
    expected_soundSpeed_HEM = data["HEM_sound_speed"]
    assert soundSpeed_HEM == pytest.approx(expected_soundSpeed_HEM, rel=1e-3)
