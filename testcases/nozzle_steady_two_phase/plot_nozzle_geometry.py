from pyshockflow.config import Config
from pyshockflow.post_processing import expansion_device_geometry_plot
import matplotlib.pyplot as plt

"""
Plotting the nozzle physical and virtual geometry incl. area distribution in order to perform a visual check of the mesh used during calculations.
"""


configFile = "inputs/config_files/CM-10.2/test_conf.ini"
config = Config(configFilePath=configFile)
fig = expansion_device_geometry_plot(config=config)
plt.show()