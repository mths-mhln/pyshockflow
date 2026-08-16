import numpy as np
import matplotlib.pyplot as plt
from numpy import pi as PI
from PyShockTube.styles import *

# INPUT
xStart = 5.0
xEnd = 6.0
D_tube = 0.3
D_throat = 0.2
D_exit = 0.4





A_tube = PI*(D_tube**2)/4
A_throat = PI*(D_throat**2)/4
A_exit = PI*(D_exit**2)/4
print(f"The area of the tube is {A_tube:.6e}.")
print(f"The area of the throat is {A_throat:.6e}.")
print(f"The area of the exit is {A_exit:.6e}.")
print(f"The ratio between throat and exit section is {A_throat/A_exit:.6e}.")
nPoints = 5000

x = np.linspace(xStart, xEnd+(xEnd-xStart)*0.1, nPoints)

xLocations = np.array([xStart, (xStart+xEnd)/2, xEnd])
areaLocations = np.array([A_tube, A_throat, A_exit])
coeffs = np.polyfit(xLocations, areaLocations, 2)
area = np.polyval(coeffs, x)


# # parabolic function with throat in the middle
# Acoeff = -4*(A_throat-A_tube)
# Bcoeff = -Acoeff
# Ccoeff = A_tube
# print(f"The equation of the nozzle is: A = {Acoeff:.6e}*z^2 + {Bcoeff:.6e}*z + {Ccoeff:.6e}")
# z = (x-x[0])/(x[-1]-x[0])
# Area = (Acoeff*z**2 + Bcoeff*z + Ccoeff)

plt.figure()
plt.plot(xLocations, areaLocations, 'o', label='Control Points')
plt.plot(x, area, label='Quadratic nozzle')
plt.xlabel('x')
plt.ylabel('A')
plt.legend()
plt.grid(alpha=.3)


print()
print("After polyfitting, the new info are these:")
print(f"The area of the tube is {A_tube:.6e}.")
print(f"The area of the throat is {np.min(area):.6e}.")
print(f"The diameter of the throat is {np.sqrt(4*np.min(area)/PI):.6e}.")
print(f"The area of the exit is {A_exit:.6e}.")
print(f"The ratio between throat and exit section is {np.min(area)/A_exit:.6e}.")


with open('nozzle.csv', 'w') as file:
    file.write('x,A\n')
    for i in range(len(x)):
        file.write('%.6e,%.6e\n' %(x[i], area[i]))

plt.show()



