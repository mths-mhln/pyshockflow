from pyshockflow import Config
from pyshockflow import Driver

from pyshockflow.visualization import plotNozzleGeometry

"""
Plotting the nozzle physical and virtual geometry incl. area distribution in order to perform a visual check of the mesh used during calculations.
"""

configFile = 'input.ini'
config = Config(configFile)
driver = Driver(config)
plotNozzleGeometry(driver)