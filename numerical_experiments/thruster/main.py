from pyshockflow import Driver
from pyshockflow import Config


config = Config('input.ini')
tube = Driver(config)
tube.solve()
