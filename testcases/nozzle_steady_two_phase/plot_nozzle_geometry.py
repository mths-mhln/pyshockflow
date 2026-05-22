import sys, os

from pyshockflow import Config
from pyshockflow import Driver

from pyshockflow.visualization import plotNozzleGeometry

"""
Plotting the nozzle physical and virtual geometry incl. area distribution in order to perform a visual check of the mesh used during calculations.
"""
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

configFile = 'inputs/input_files/lettieri/input_HEOS_CoolProp_lettieri_L5.ini'
config = Config(configFile)
with HiddenPrints():
    driver = Driver(config)
plotNozzleGeometry(driver)