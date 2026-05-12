from pyshockflow.visualization import make_animations

# Input data
picklePath = 'Results.pik' # Path to the pickle file of the simulation
maxLength = 100 # choose how many snapshots you want to visualize (must be < than total snapshots of simulation)

# video settings
FPS = 30 # frames per second
DPI = 400 # definition (<500 works, more no, don't know why)

make_animations(
    picklePath = picklePath, 
    maxLength = maxLength, 
    FPS = FPS, 
    DPI = DPI)

