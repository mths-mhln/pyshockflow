# Conceptual Heat Pump Cycle (CHPC)

This tool identifies the vapour-compression cycle with maximum COP for a given set of heat source and heat sink characteristics, heat exchanger pinch-point constraints, component efficiencies and refrigerant. The cycle is a single-stage vapour-compression cycle in which the expansion valve is replaced by a turbine expander. It follows from a differential evolution optimization of the compressor pressure ratio, the temperature and entropy at the compressor inlet, and the temperature at the condenser outlet, subject to the pinch-point constraints of the evaporator and condenser. The figure below shows the resulting cycle of R1234ze(E), together with the expansion through an isentropic turbine and through an expansion valve. The diagram background is generated with [thermoplot](https://github.com/mths-mhln/thermoplot), of which a local copy is included in `src/thermoplot`.

The documentation can be found at [mths-mhln.github.io/CHPCT](https://mths-mhln.github.io/CHPCT/).

![Conceptual heat pump cycle of R1234ze(E) in a T-s diagram](images/hp_cycle_TS.png)

## Dependencies

- Python 3.13 or newer
- CoolProp, matplotlib, numpy, rich and scipy, see `pyproject.toml`. The exact versions are recorded in `uv.lock`.
- NIST REFPROP. All thermodynamic properties are extracted from REFPROP through CoolProp, hence a REFPROP licence and installation are necessary.
- A LaTeX installation, which is used for the text in the diagrams.

## Installation

Clone the repository, create a virtual environment with [uv](https://docs.astral.sh/uv/) and install the tool in editable mode, which makes both `CHPC` and the local copy of `thermoplot` importable:

```bash
git clone https://github.com/mths-mhln/CHPCT.git
cd CHPCT
uv venv
.venv\Scripts\activate
uv pip install -e .
```

On Linux and macOS the environment is activated with `source .venv/bin/activate`.

## Source structure

The Python package is organized by responsibility:

| Folder | Contents |
| --- | --- |
| `src/CHPC/cycle_evaluation/` | Thermodynamic cycle solvers and cycle data generation |
| `src/CHPC/post_processing/` | Diagrams and post-processing studies, including the global optimum, perturbation and calculation-method comparisons |
| `src/CHPC/utils/` | Configuration parsing, result serialization and logging |
| `src/CHPC/COP_vs_eff/` | COP versus component-efficiency sweep |

## How to run

Running the tool and post-processing its results are separated. `main.py` runs the analysis specified in the config file at the top of the script, and saves the results as `.npz` files in `results/`. Afterwards, `postprocess.py` makes the diagrams from these files:

```bash
python main.py
python postprocess.py
```

Before the analysis, `main.py` can verify the model against pocketTHERM for three heat pump cycles (see `test/system_tests`), which is switched on with `perform_verification=True` at the bottom of `main.py`. The refrigerant folders in `config/` contain two kinds of config files:

| Config file | Analysis | Results |
| --- | --- | --- |
| `CHPC.ini` | optimizes a single cycle and prints its performance parameters, including the volumetric expansion ratio over the turbine | T-s, p-h and optimizer progress diagrams in `results/heat_pump_thermodynamic_diagrams/<refrigerant>/` |
| `COP_vs_eff.ini` | optimizes the cycle for a grid of turbine and compressor efficiencies | COP maps in `results/COP_investigations/<refrigerant>/` |

The COP sweep saves its progress every 5 cycles, and can be continued after an interruption. The figure below shows the COP map of R1234ze(E).

![COP of R1234ze(E) as a function of the turbine and compressor efficiency](images/COP_vs_eff.png)

The cycle calculation method is selected with `calculation_method`. `full_optimization` optimizes the four variables above, `fixed_PR` optimizes the compressor inlet state and the condenser outlet temperature at a given pressure ratio, and `three_temp_fix` fixes four temperature differences and only optimizes the pressure ratio (the implementation is in `src/CHPC/cycle_evaluation/thermodynamics_three_temp_fix.py`). Results of the latter two methods carry the method in their file name, e.g. `COP_vs_eff_R1234ze(E)_COP_turb_three_temp_fix.npz`, such that the methods can be compared with

```bash
python -m CHPC.post_processing.calculation_method_comparison --data-dir "results/COP_investigations/R1234ze(E)/data"
```

Two investigations support the results of the thesis, and are run as modules from the root of the repository. `python -m CHPC.post_processing.global_optimum_check` repeats the optimization of a cycle for several seeds of the differential evolution algorithm, and shows the coverage of the design space by the evaluated candidates. `python -m CHPC.post_processing.perturbation_study` perturbs the cycle specifications of the R1234ze(E), CO2 and MM config files one at a time by -20 % and +20 %, and evaluates the optimal cycle of every configuration in parallel.

`postprocess.py` also contains the empty T-s and p-h diagrams of a substance. The margins around the cycle and the size of the optimizer progress figure are arguments of `make_cycle_plots`, and the diagram backgrounds are specified in `config/thermoplot/`.

## Input overview

The config file is an `.ini` file with the sections `[ANALYSIS TYPE]`, `[ANALYSIS SPECIFICATIONS]`, `[COP VS EFF SETTINGS]` (`COP_vs_eff` only) and `[NUMERICS]`, see `config/R1234ze(E)/CHPC.ini` for an example. Inputs that are not used by the specified analysis type and calculation method must not be specified, e.g. `PR` for `full_optimization`. An overview of all inputs is given in the [documentation](https://mths-mhln.github.io/CHPCT/inputs.html).
