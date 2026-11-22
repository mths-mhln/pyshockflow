# PyShockFlow

PyShockFlow solves the quasi-1D Euler equations for the flow of an ideal gas or a real fluid through a duct of varying cross-section. It was originally developed for shock tubes, and has been extended to the steady flow through converging-diverging nozzles, including flashing and condensing two-phase flows. In the two-phase region, the equations are closed by the homogeneous equilibrium model (HEM), with thermophysical properties from NIST REFPROP through CoolProp. The equations are discretized with a finite-volume method, Roe-type numerical fluxes and explicit time marching to the steady state.

The documentation is in `docs/`, and covers the installation, running a simulation, all inputs of the configuration file, the model and the post-processing. It is built with Sphinx:

```bash
uv pip install -r docs/requirements.txt
cd docs
make html
```

after which the pages are found in `docs/build/html/index.html`.

## Installation

The tool requires Python 3.12 or newer. Clone the repository, create a virtual environment with [uv](https://docs.astral.sh/uv/) and install the packages in editable mode:

```bash
git clone https://github.com/mths-mhln/pyshockflow.git
cd pyshockflow
uv venv
.venv\Scripts\activate
uv pip install -e .
```

On Linux and macOS the environment is activated with `source .venv/bin/activate`. Real-fluid and two-phase simulations with REFPROP require a REFPROP licence and installation, in a location where CoolProp can find it.

The installation is checked by running the automated tests from the root folder:

```bash
uv run pytest test/regression_tests/test_cases.py
uv run pytest test/integration_tests/test_fluid.py
```

## How to run

A simulation is defined by a configuration file (`.ini`) and a geometry file (`.csv`), and is run from a short script in the folder that contains them:

```python
from pyshockflow import Config, Driver

config = Config(configFilePath="inputs/config_files/lettieri/L1_smooth.ini")
driver = Driver(config=config)
driver.solve()
```

The results are written to `Results/<RESULTS_SUBDIRECTORY_NAME>/Results.pik`, and are plotted with the functions in `pyshockflow.post_processing`. The folders in `test/regression_tests/` are the smallest complete examples, and `numerical_experiments/nozzle_steady_two_phase/` contains the two-phase nozzle cases, together with `main.py` and `postprocess.py`.

## Contribution guidelines

- Verify modifications with detailed test cases.
- Make sure that the regression tests still pass, and extend them if needed.

## Authors

- Francesco Neri, TU Delft, original PyShockFlow
- Matteo Pini, TU Delft, original PyShockFlow
- Matthias Van Mechelen, nozzle and two-phase extensions (MSc thesis *Towards Turbine-Induced Expansion in High-Temperature Heat Pumps*, TU Delft, 2026)

## References

[1] E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, Springer, 2009.

[2] S. Arabi, J.-Y. Trépanier and R. Camarero, "A simple extension of Roe's scheme for real gases", *Journal of Computational Physics* 329, 16-28, 2017.

[3] A. Cioffi, A. Bergamin, R. Agromayor and F. Haglind, "A hyperbolic one-dimensional model for two-phase flows in converging-diverging nozzles", *Journal of Turbomachinery* 148(3), 031017, 2026.

[4] S. D'Alessandro, M. Pizzarelli and F. Nasuti, "A hybrid real/ideal gas mixture computational framework to capture wave propagation in liquid rocket combustion chamber conditions", *Aerospace* 8(9), 250, 2021.
