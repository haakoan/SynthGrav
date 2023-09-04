import numpy as np
"""
Functions that define different modes.
The formulas are taken from 

Universal relations for gravitational-wave asteroseismology of proto-neutron stars
Torres-Forné et al 2021
https://arxiv.org/abs/1902.10048

Note on the naming system for frequency functions:
All modes are of the second order of the spherical harmonic decomposition
(l=2). The number following p or g in the functions calculating frequencies
associated with these modes indicates the number of radial nodes.
"""

def fmode(**kwargs):
    """
    Generates an f-mode gravitational wave frequency based on stellar mass and radius.

    This function takes the dimensionless mass and radius of the star and calculates the f-mode frequency
    using the relationship: f = b * x + c * x ** 2, where x = np.sqrt(msh / rsh ** 3).

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing the following key-value pairs:
            msh : numpy.array
                An array of the same length as `time` containing the dimensionless mass values (msh) of the star.
            rsh : numpy.array
                An array of the same length as `time` containing the dimensionless radius values (rsh) of the star.

    Returns
    -------
    numpy.array
        An array containing the f-mode gravitational wave frequencies. The output is normalized to 1.

    Notes
    -----
    - The f-mode frequency depends on the mass (msh) and radius (rsh) of the star.
    - The coefficients b and c are determined empirically.
    - Parameters `msh` and `rsh` must be passed through the argument **kwargs as numpy arrays with the same length as signal.time.
    """
    msh = kwargs["msh"]
    rsh = kwargs["rsh"]
    b = 1.410e5
    c = -4.23e6

    x = np.sqrt(msh / rsh ** 3)
    f = b * x + c * x ** 2
    return f


def p1mode(**kwargs):
    """
    Generates p1-mode gravitational wave frequency based on stellar mass and radius.

    This function takes the dimensionless mass and radius of the star and calculates the p1-mode frequency
    using the relationship: f = b * x + c * x ** 2, where x = np.sqrt(msh / rsh ** 3).

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing the following key-value pairs:
            msh : numpy.array
                An array of the same length as `time` containing the dimensionless mass values (msh) of the star.
            rsh : numpy.array
                An array of the same length as `time` containing the dimensionless radius values (rsh) of the star.

    Returns
    -------
    numpy.array
        An array containing the p1-mode gravitational wave frequencies. The output is normalized to 1.

    Notes
    -----
    - Parameters `msh` and `rsh` must be passed through the argument **kwargs as numpy arrays with the same length as signal.time.
    """
    msh = kwargs["msh"]
    rsh = kwargs["rsh"]

    b = 2.205e5
    c = 4.63e6

    x = np.sqrt(msh / rsh ** 3)
    f = b * x + c * x ** 2
    return f

def p2mode(**kwargs):
    """
    Generates p2-mode gravitational wave frequency based on stellar mass and radius.

    This function takes the dimensionless mass and radius of the star and calculates the p2-mode frequency
    using the relationship: f = b * x + c * x ** 2, where x = np.sqrt(msh / rsh ** 3).

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing the following key-value pairs:
            msh : numpy.array
                An array of the same length as `time` containing the dimensionless mass values (msh) of the star.
            rsh : numpy.array
                An array of the same length as `time` containing the dimensionless radius values (rsh) of the star.

    Returns
    -------
    numpy.array
        An array containing the p2-mode gravitational wave frequencies. The output is normalized to 1.

    Notes
    -----
    - Parameters `msh` and `rsh` must be passed through the argument **kwargs as numpy arrays with the same length as signal.time.
    """
    msh = kwargs["msh"]
    rsh = kwargs["rsh"]

    b = 4.02e5
    c = 7.4e6

    x = np.sqrt(msh / rsh ** 3)
    f = b * x + c * x ** 2
    return f

def p3mode(**kwargs):
    """
    Generates p3-mode gravitational wave frequency based on stellar mass and radius.

    This function takes the mass (in solar masses) and radius (in km) of the star and calculates the p3-mode frequency
    using the relationship: f = b * x + c * x ** 2, where x = np.sqrt(msh / rsh ** 3).

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing the following key-value pairs:
            msh : numpy.array
                An array of the same length as `time` containing the mass values (in solar masses) of the star.
            rsh : numpy.array
                An array of the same length as `time` containing the radius values (in km) of the star.

    Returns
    -------
    numpy.array
        An array containing the p3-mode gravitational wave frequencies. The output is normalized to 1.

    Notes
    -----
    - Parameters `msh` and `rsh` must be passed through the argument **kwargs as numpy arrays with the same length as signal.time.
    """
    msh = kwargs["msh"]
    rsh = kwargs["rsh"]

    b = 6.21e5
    c = -1.9e6

    x = np.sqrt(msh / rsh ** 3)
    f = b * x + c * x ** 2
    return f

def g1mode(**kwargs):
    """
    Generates g1-mode gravitational wave frequency based on stellar mass and radius.

    This function takes the mass (in solar masses) and radius (in km) of the star and calculates the g1-mode frequency
    using the relationship: f = b * x + c * x ** 2, where x = M_pns / R_pns ** 2.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing the following key-value pairs:
            mpns : numpy.array
                An array of the same length as `time` containing the mass values (in solar masses) of the star.
            rpns : numpy.array
                An array of the same length as `time` containing the radius values (in km) of the star.

    Returns
    -------
    numpy.array
        An array containing the g1-mode gravitational wave frequencies. The output is normalized to 1.

    Notes
    -----
    - Parameters `mpns` and `rpns` must be passed through the argument **kwargs as numpy arrays with the same length as signal.time.
    """
    mpns = kwargs["mpns"]
    rpns = kwargs["rpns"]

    b = 8.67e5
    c = -51.9e6

    x = mpns / rpns ** 2
    f = b * x + c * x ** 2
    return f


def g2mode(**kwargs):
    """
    Generates g2-mode gravitational wave frequency based on stellar mass and radius.

    This function takes the mass (in solar masses) and radius (in km) of the star and calculates the g2-mode frequency
    using the relationship: f = b * x + c * x ** 2 + d * x ** 3, where x = M_pns / R_pns ** 2.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing the following key-value pairs:
            mpns : numpy.array
                An array of the same length as `time` containing the mass values (in solar masses) of the star.
            rpns : numpy.array
                An array of the same length as `time` containing the radius values (in km) of the star.

    Returns
    -------
    numpy.array
        An array containing the g2-mode gravitational wave frequencies. The output is normalized to 1.

    Notes
    -----
    - Parameters `mpns` and `rpns` must be passed through the argument **kwargs as numpy arrays with the same length as signal.time.
    """
    mpns = kwargs["mpns"]
    rpns = kwargs["rpns"]

    b = 5.88e5
    c = -86.2e6
    d = 4.67e9

    x = mpns / rpns ** 2
    f = b * x + c * x ** 2 + d * x ** 3
    return f

def g3mode(**kwargs):
    """
    Generates g3-mode gravitational wave frequency based on stellar mass, radius, central pressure, and central density.

    This function takes the mass (in solar masses), radius (in km), central pressure, and central density of the star 
    and calculates the g3-mode frequency using the relationship: f = a + b * x + c * x ** 2, where 
    x = np.sqrt(msh / rsh ** 3) * (pC / rhoC ** 2.5).

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing the following key-value pairs:
            msh : numpy.array
                An array of the same length as `time` containing the mass values (in solar masses) of the star.
            rsh : numpy.array
                An array of the same length as `time` containing the radius values (in km) of the star.
            pC : numpy.array
                An array of the same length as `time` containing the central pressure values of the star.
            rhoC : numpy.array
                An array of the same length as `time` containing the central density values of the star.

    Returns
    -------
    numpy.array
        An array containing the g3-mode gravitational wave frequencies. The output is normalized to 1.

    Notes
    -----
    - Parameters `msh`, `rsh`, `pC`, and `rhoC` must be passed through the argument **kwargs as numpy arrays with the same length as signal.time.
    """
    msh = kwargs["msh"]
    rsh = kwargs["rsh"]
    pC = kwargs["pC"]
    rhoC = kwargs["rhoC"]

    a = 905
    b = -79.9e5
    c = -11000e6

    x = np.sqrt(msh / rsh ** 3) * (pC / rhoC ** 2.5)
    f = a + b * x + c * x ** 2
    return f

def gmode_default_func(**kwargs):
    """
    A simple toy mode function that generates a linearly increasing frequency over time.

    This function takes the time array and returns a frequency array with a linear relationship: f = 160 + 1400 * t.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing the following key-value pair:
            time : numpy.array
                An array containing the time values for which the frequency will be calculated.

    Returns
    -------
    numpy.array
        An array containing the gravitational wave frequencies generated by the simple toy mode function.

    Notes
    -----
    - Parameter `time` must be passed through the argument **kwargs as a numpy array.
    """
    t = kwargs['time']
    return 160 + 1400*t


builtin_modes = {"fmode" : fmode, "p1mode" : p1mode, "p2mode" : p2mode, "p3mode" : p3mode,
                 "g1mode": g1mode, "g2mode": g2mode, "g3mode": g3mode, "gmode_default_func" : gmode_default_func}
