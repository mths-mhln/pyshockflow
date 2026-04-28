import pickle 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os    

#########################     INPUT         #########################

picklePath = 'Results.pik' # Path to the pickle file of the simulation
maxLength = 100 # choose how many snapshots you want to visualize (must be < than total snapshots of simulation)

#video settings
FPS = 30 # frames per second
DPI = 400 # definition (<500 works, more no, don't know why)

#####################################################################




# open the pickle file
with open(picklePath, 'rb') as file:
    solution = pickle.load(file)

# save aliases for the arrays of interest
x = solution['X Coords']
time = solution['Time']
rho = solution['Primitive']['Density']
u = solution['Primitive']['Velocity']
p = solution['Primitive']['Pressure']
e = solution['Fluid'].computeStaticEnergy_p_rho(p, rho)
nPoints, nTimes = rho.shape
iterations = np.linspace(0, nTimes-1, num=maxLength, dtype=int)

fields = [rho, u, p, e]
labels = ['Density [kg/m3]', 'Velocity [m/s]', 'Pressure [Pa]', 'Energy [J]' ]
videoNames = ['Density.mp4', 'Velocity.mp4', 'Pressure.mp4', 'Energy.mp4']

# PLOTS AND VIDEO
def plot_limits(f, extension=0.05):
    max = f.max()
    min = f.min()
    left = min-(max-min)*extension
    right = max+(max-min)*extension
    return left, right


for i,field in enumerate(fields):
    xmin, xmax = plot_limits(x)
    ymin, ymax = plot_limits(field)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], '-C0')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(labels[i])
    ax.grid(alpha=0.2)

    def update(iteration):
        line.set_data(x, field[:, iteration])
        ax.set_title(f'Time: {time[iteration]:.3e} [s]')
        fig.tight_layout()
        return line,

    ani = animation.FuncAnimation(fig, update, frames=iterations, blit=False)

    # Save the animation as a video
    ani.save(videoNames[i], writer='ffmpeg', fps=FPS, dpi=DPI)
    print('Video %s Saved' %(videoNames[i]))

# plt.show()
