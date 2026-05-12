import pickle 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def make_animations(picklePath, maxLength, FPS, DPI):
    """
    Show the temporal evolution of the solution over the computational comain. 

    Arguments
    ---------
    picklePath : str
        The path to the pickle file of the simulation. Contains dictionary of 2D np arrays for each primitive variable, (space, time)
    maxLength : int
        The maximum number of time steps to include in the animation.
    FPS : int
        The frames per second for the animation.
    DPI : int
        The dots per inch for the saved video.

    Returns
    -------
    None. Saves the videos in the directory from which the file is executed.
    """
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

    return None



def plotNozzleGeometry(Driver):
        """
        Plot the nozzle geometry and numerical grid.

        Arguments
        ---------
        Driver : Driver
            The Driver object capable of extracting the nozzle geometry and constructing the 1D mesh

        Returns
        -------
        None. Saves the plot in the directory from which the file is executed.
        """
        nozzleData = np.loadtxt(Driver.config.getNozzleFilePath(), skiprows=1, delimiter=',', dtype=float)
        nozzleX = nozzleData[:,0]
        nozzleArea = nozzleData[:,1]

        # Linear interpolation with external filling set to area Reference (=Tube area)
        interpolatedNozzleArea = np.interp(Driver.xNodesVirtual, nozzleX, nozzleArea, left=nozzleData[0,1], right=nozzleData[-1,1])

        # Scale plot axes according to the nozzle geometry
        x_scale = Driver.xNodesVirtual[-1]
        area_scale = 2*max(interpolatedNozzleArea)
        ratio = area_scale / x_scale
        
        fig = plt.figure(figsize=(12, 12*ratio))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(Driver.xNodesVirtual, interpolatedNozzleArea, label='Interpolated Nozzle Area', color='blue')
        ax.plot(Driver.xNodesVirtual, -interpolatedNozzleArea, label='Interpolated Nozzle Area', color='blue')
        ax.scatter(Driver.xNodesVirtual, np.zeros_like(Driver.xNodesVirtual), color='red', label='Virtual Mesh Nodes', s=0.5)
        ax.set_xlabel('x [m]', fontsize=12)
        ax.set_ylabel('Area [m^2]', fontsize=12)
        ax.set_title('Nozzle Geometry', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        fig.show()
        fig.tight_layout()
        fig.savefig("nozzle_geometry.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.show()
        return None