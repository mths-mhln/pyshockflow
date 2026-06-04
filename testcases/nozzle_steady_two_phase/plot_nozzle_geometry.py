from pyshockflow import Config
from pyshockflow import Driver

import pyshockflow.post_processing
from pyshockflow.post_processing import nozzle_geometry_plot
import matplotlib.pyplot as plt

"""
Plotting the nozzle physical and virtual geometry incl. area distribution in order to perform a visual check of the mesh used during calculations.
"""


configFile = 'inputs/input_files/orchid/input_recovery.ini'
config = Config(configFile)
with pyshockflow.post_processing.HiddenPrints():
    driver = Driver(config)
fig = nozzle_geometry_plot(driver)
plt.show()