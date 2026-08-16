
from pyshockflow.post_processing import expansion_device_geometry_plot
import matplotlib.pyplot as plt

"""
Plotting the nozzle physical and virtual geometry incl. area distribution in order to perform a visual check of the mesh used during calculations.
"""


configFile = 'inputs/config_files/orchid/testcase.ini'
fig = expansion_device_geometry_plot(configFilePath=configFile)
plt.show()