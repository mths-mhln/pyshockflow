API reference
=============

For most users, the configuration file and the three objects below are the only interface to the tool. The reference is generated from the docstrings in the code.

Configuration and solver
------------------------

``Config`` reads a configuration file and checks it against the rules of :ref:`input-rules`. The values are accessed through one method per key, e.g. ``config.numberOfMeshNodes()`` or ``config.inletConditionsValues()``.

.. autoclass:: pyshockflow.Config

.. autoclass:: pyshockflow.Driver
   :members: solve

Post-processing
---------------

.. autoclass:: pyshockflow.post_processing.VandVSpec

.. autofunction:: pyshockflow.post_processing.expansion_device_geometry_plot

.. autofunction:: pyshockflow.post_processing.generate_fluid_state_var_profile_figs

.. autofunction:: pyshockflow.post_processing.compute_v_and_v_metrics

.. autofunction:: pyshockflow.post_processing.generate_expansion_thermoplot

.. autofunction:: pyshockflow.post_processing.render_profile_video

.. autofunction:: pyshockflow.post_processing.use_plot_style
