import sys
import pickle
import pytest

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from scipy.optimize import newton

from pyshockflow.fluid import FluidReal
import fluid_properties.fluid_properties as FP
from fluid_properties.coolprop_interface import CoolPropAbstractState_v2

@pytest.fixture
def fluid_real_obj(request):
    fluid_name = request.param
    return FluidReal(fluid_name, "REFPROP", "abstractstate_v2")

@pytest.mark.parametrize(
    ("fluid_real_obj", "verification_data_path"),
    (
        ("CO2", "data/test_fluid/CM56_SOS_testing_test_1_CO2.pkl"),
        ("R1234ze(E)", "data/test_fluid/CM56_SOS_testing_test_1_R1234ze(E).pkl"),
    ),
    # Don't pass this value straight into the test function. Instead, find the fixture
    # called fluid_real_obj, and feed this value into that fixture as request.param
    indirect=["fluid_real_obj"]
)
def test_CM56_SOS_testing_test_1(fluid_real_obj, verification_data_path):
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
    For more info, see the script test/tests_nozzle_steady_two_phase/integration_tests/scripts/generate_data_test_fluid/generate_data_test_fluid.py
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
    (("Water", "data/test_fluid/CM56_SOS_testing_test_2.pkl"),),
    indirect=["fluid_real_obj"],
)
def test_CM56_SOS_testing_test_2(fluid_real_obj, verification_data_path):
    """
    Verification of computeSoundSpeed_p_rho method of the FluidReal class against figure 1b 
    of Marco De Lorenzo "Benchmark of the Delayed Equilibrium Model". Pertains variation of 
    HEM sound speed with volume fraction of vapour for pressure 0.1 MPa and water as working
    fluid. For more info, see the script test/tests_nozzle_steady_two_phase/integration_
    tests/scripts/generate_data_test_fluid/generate_data_test_fluid.py
    """
    with open(verification_data_path, "rb") as f:
        data = pickle.load(f)
    
    # data structure
    # {"alpha_V": np.1darray([...]), "HEM_sound_speed": np.1darray([...])}

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



@pytest.mark.parametrize(
    ("fluid_real_obj", "verification_data_path"),
    (("Water", "data/test_fluid/CM56_SOS_testing_test_3.pkl"),),
    indirect=["fluid_real_obj"],
)
def test_CM56_SOS_testing_test_3(fluid_real_obj, verification_data_path):
    """
    Verification of computeSoundSpeed_p_rho method of the FluidReal class against the HEM
    SOS equation presented in Marco De Lorenzo "Benchmark of the Delayed Equilibrium Model".
    This formulation differs from the Cioffi et al. formulation implemented in the FluidReal.
    class. 
    """
    # extract tests data generated in scripts/generate_data_test_fluid/generate_data_test_fluid.py
    # for CM56_SOS_testing_test_3
    with open(verification_data_path, "rb") as f:
        data = pickle.load(f)
    
    # data structure
    # {"p": np.1darray([...]), "rho": np.1darray([...]), "soundSpeed_HEM_de_lorenzo": np.1darray([...])}
    p = data["p"]
    rho = data["rho"]
    expected_soundSpeed_HEM_de_lorenzo = data["soundSpeed_HEM_de_lorenzo"]

    # compute sound speed using the FluidReal method (Cioffi et al. A Hyperbolic One-Dimensional Model 
    # for Two-Phase Flows in Converging-Diverging Nozzles)
    soundSpeed_HEM = fluid_real_obj.computeSoundSpeed_p_rho(p, rho)

    # compare if the two formulations agree, increasing confidence in correct implementation
    # of the Cioffi formulation.
    assert soundSpeed_HEM == pytest.approx(expected_soundSpeed_HEM_de_lorenzo, rel=1e-3)



@pytest.mark.parametrize(
    ("fluid_real_obj", "verification_data_path"),
    (("R1234ze(E)", "data/test_fluid/CM92_CoolProp_mixture_thdy_properties.pkl"),),
    indirect=["fluid_real_obj"],
)
def test_CM92_CoolProp_mixture_thdy_properties(fluid_real_obj, verification_data_path):
    """
    Verification of the CoolProp mixture properties against the values obtained 
    when solving the Giljarhus system. For a more comprehensive description of the 
    test, please refer to the script test/tests_nozzle_steady_two_phase/integration_
    tests/scripts/generate_data_test_fluid/generate_data_test_fluid.py
    """

    # extract test data generated in scripts/generate_data_test_fluid/generate_data_test_fluid.py
    with open(verification_data_path, "rb") as f:
        data = pickle.load(f)

    # data structure
    # {"D": {"vals": np.2darray([...]), "coords": (S_grid, T_grid)},
    #  "U": {"vals": np.2darray([...]), "coords": (S_grid, T_grid)},
    #  "P": {"vals": np.2darray([...]), "coords": (S_grid, T_grid)},
    #  "Q": {"vals": np.2darray([...]), "coords": (S_grid, T_grid)}}

    AS = CoolPropAbstractState_v2(fluid_real_obj.fluid.Library, fluid_real_obj.fluid.Name)

    for prop_type in data.keys():
        # extract coords (at which to evaluate the property)
        S_eval, T_eval = data[prop_type]["coords"]

        # compute the property using the FluidReal class
        prop_eval = AS.PropsSI(prop_type, "T", T_eval.ravel(), "S", S_eval.ravel(), verbose=False)
        # Normally, methods of the fluid_real object must be used for proper testing. 
        # In my opinion, use of the fluid_real object is quite silly. It is a very roundabout way of 
        # calling the FP class PropsSI method, which already generalizes. The problem:
        # the fluid_real object does not have methods to evaluate properties from S, T inputs
        # for all variables of interest during this investigation, hence for me to test, 
        # new methods must be made. Instead I just used the AS method I created. This does not generalize to 
        # all fluid-dynamic libraries and may have to be extended in the future, but for now 
        # the two-phase flow implementation only works for CoolProp, so this is sufficient.
        # I use the AS object due to it's verbose option, which suppresses the CoolProp warnings.

        # drop all NaN values, for assertion testing.
        prop_eval = prop_eval[~np.isnan(prop_eval)]    
        print(np.sum(np.isnan(prop_eval)))

        # compare the computed property with the expected property from the verification data
        expected_prop_eval = data[prop_type]["vals"].ravel()
        print(np.sum(np.isnan(expected_prop_eval)))

        print(prop_eval.size, expected_prop_eval.size)
        assert prop_eval == pytest.approx(expected_prop_eval, rel=1e-3)
