###########################################
# Imports
###########################################
import matplotlib.pyplot as plt
import numpy as np

from configthermoplot import ConfigThermoplot
from coolprop_interface_thermoplot import CoolPropAbstractState





def extract_critical_point(config: type[ConfigThermoplot], AS: type[CoolPropAbstractState]) -> np.ndarray:
    """extract correct critical point coordinates according to diagram type. """
    # Only T_crit, P_crit and rho_crit can be used for extracting critical point coordinates. 
    # use rho_crit as second variable for extracting S and H information. 
    iv_type = config.thermoplot_settings["diagram_type"][0]
    dv_type = config.thermoplot_settings["diagram_type"][-1]
    iv_crit = AS.PropsSI(f"{iv_type}crit")
    rho_crit = AS.PropsSI("Dcrit")
    dv_crit = AS.PropsSI(dv_type, "D", rho_crit, iv_type, iv_crit)

    # convert to np.ndarray for easy appending to isoline data
    crit_coords = np.array([[dv_crit, iv_crit]])

    return crit_coords



def configure_matplotlib():
    # give matplotlib a LaTeX formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Utopia"],
        "text.latex.preamble": (
            r"\usepackage[T1]{fontenc}"
            r"\usepackage{lmodern}"
            r"\usepackage{siunitx}"
            r"\usepackage{bm}"
            r"\sisetup{group-separator={\,},group-minimum-digits=4}"
        ),
    })
    plt.rcParams['path.simplify'] = True
    plt.rcParams['path.simplify_threshold'] = 0.1
    return None