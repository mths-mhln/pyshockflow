from pyshockflow import Driver
from pyshockflow import Config


nameList = ['1stOrder', 'minmod', 'vanalbada', 'vanleer', 'superbee']
inputList = ['input_%s.ini' %(name) for name in nameList]

for input in inputList:
    config = Config(input)
    tube = Driver(config)
    tube.solve()
