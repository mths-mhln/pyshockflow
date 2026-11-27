Installation
============

Requirements
------------

- Python 3.12 or newer.
- The packages listed in ``pyproject.toml``, among which CoolProp 6.8.0, NumPy, SciPy, Matplotlib and Pandas. The exact versions are recorded in ``uv.lock``. NumPy is pinned to 2.2.1, since the post-processing still relies on ``numpy.compat``.
- NIST REFPROP, for real-fluid and two-phase simulations. REFPROP is licensed software and must be installed separately, in a location where CoolProp can find it. On Windows, the default installation folder is found automatically; otherwise the path is set with ``CoolProp.CoolProp.set_config_string(CoolProp.CoolProp.ALTERNATIVE_REFPROP_PATH, path)``. Ideal-gas simulations, and real-fluid simulations with the HEOS backend of CoolProp, do not need REFPROP, but the T-s diagrams of the post-processing do.
- Optionally FFmpeg, either as a system installation or through the ``imageio-ffmpeg`` package. Videos are written as MP4 files if FFmpeg is available, and as GIF files otherwise.

Installing the tool
-------------------

Clone the repository, create a virtual environment with `uv <https://docs.astral.sh/uv/>`_ and install the packages in editable mode:

.. code-block:: bash

   git clone https://github.com/mths-mhln/pyshockflow.git
   cd pyshockflow
   uv venv
   .venv\Scripts\activate
   uv pip install -e .

On Linux and macOS, the environment is activated with ``source .venv/bin/activate``. The editable installation makes four packages importable from anywhere:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Package
     - Contents
   * - ``pyshockflow``
     - the solver: configuration file and input checks (``config.py``), pre-processing, boundary conditions, source terms, solver loop, output and restart (``driver.py``), ideal-gas and real-fluid models with the HEM speed of sound and viscosity (``fluid.py``), Roe-type fluxes (``advection_roe.py``), the exact Riemann solver of the Godunov scheme (``riemann_problem.py``), analytical reference solutions (``analytical_solutions.py``) and the post-processing (``post_processing.py``)
   * - ``fluid_properties``
     - a common interface to CoolProp (including REFPROP), FluidProp, look-up tables and FeOs
   * - ``thermoplot``
     - T-s and p-h diagrams, used for the expansion paths
   * - ``benchmarking_tools``
     - timing of simulations

Besides ``src/``, the repository contains ``numerical_experiments/``, with the example and thesis cases (:ref:`example-cases`), and ``test/``, with the automated tests.

Checking the installation
-------------------------

The installation is checked by running the regression and integration tests from the root folder of the repository:

.. code-block:: bash

   uv run pytest test/regression_tests/test_cases.py
   uv run pytest test/integration_tests/test_fluid.py

The regression tests rerun a number of shock-tube and nozzle cases and compare the results with stored ones. The integration tests compare the fluid model, among which the HEM speed of sound, with independent reference data, and require REFPROP.

Building this documentation
---------------------------

The documentation is written with `Sphinx <https://www.sphinx-doc.org>`_ and is built from the ``docs`` folder:

.. code-block:: bash

   uv pip install -r docs/requirements.txt
   cd docs
   make html

The pages are written to ``docs/build/html``. On Windows, ``make.bat html`` is used instead of ``make html``.
