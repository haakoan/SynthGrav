from functools import singledispatch
from collections import abc
import numpy as np
from scipy.stats import logistic, cauchy


def gauss_psd(f,central_frequency,**kwargs):
    """
    Returns the gaussian spectral density centered
    around central_frequency with standard deviation sigma.

    This function is used as a default weighting function.
    """
    sigma = kwargs['sigma']
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (f-central_frequency)**2 / (2 * sigma**2))



# Constant PSD
def constant_psd(f, central_frequency, **kwargs):
    """
    Constant Power Spectral Density function. Returns a constant value within a band defined around the central_frequency
    and zero outside the band.
    
    Args:
        f (array): Frequency array.
        central_frequency (float): The central frequency of the constant band.
        kwargs: Additional arguments, can include 'constant_value' (the constant PSD value), 'delta_f' (the width of the constant band).

    Returns:
        array: Constant PSD within the specified band and zero outside.
    """
    constant_value = kwargs.get('constant_value', 1)
    delta_f = kwargs.get('delta_f', 1)
    psd = np.where(np.abs(f - central_frequency) <= delta_f / 2, constant_value, 0)

    return psd

# Logistic PSD
def logistic_psd(f, central_frequency, **kwargs):
    """
    Logistic Power Spectral Density function.
    
    Args:
        f (array): Frequency array.
        central_frequency (float): The central frequency of the Logistic distribution.
        kwargs: Additional arguments, should include 'mu' (the mean of the Logistic distribution) and 's' (scale parameter of the Logistic distribution).

    Returns:
        array: Logistic PSD.
    """
    mu = kwargs['mu']
    s = kwargs['s']
    pdf = logistic.pdf(f - central_frequency, mu, s)
    return pdf / np.sum(pdf)

# Cauchy PSD
def cauchy_psd(f, central_frequency, **kwargs):
    """
    Cauchy Power Spectral Density function.
    
    Args:
        f (array): Frequency array.
        central_frequency (float): The central frequency of the Cauchy distribution.
        kwargs: Additional arguments, should include 'x0' (the location parameter or median of the Cauchy distribution) and 'gamma' (the scale parameter or half width at half maximum of the Cauchy distribution).

    Returns:
        array: Cauchy PSD.
    """
    x0 = kwargs['x0']
    gamma = kwargs['gamma']
    pdf = cauchy.pdf(f - central_frequency, x0, gamma)
    return pdf / np.sum(pdf)


def colored_noise_psd(f, **kwargs):
    """
    Colored Noise Power Spectral Density function.

    This function generates a PSD based on the exponent of the frequency array (f**beta). 
    It can represent different types of noise depending on the value of beta.

    White Noise: beta = 0
    Blue Noise: beta = 0.5
    Violet Noise: beta = 1
    Brownian Noise: beta = -2
    Pink Noise: beta = -0.5

    Args:
        f (array): Frequency array.
        kwargs: Additional arguments, should include 'beta' (the exponent applied to the frequency array).

    Returns:
        array: PSD values based on the specified exponent of the frequency.
    """
    beta = kwargs.get('beta', 1)  # Default value of beta is 1

    # Handling division by zero when beta < 0
    if beta < 0:
        # Avoid division by zero
        f = np.where(f == 0, float('inf'), f)
    
    return f**beta



def gen_pulse(N, dt, psd, central_frequency, rng_generator, 
              polarisation='unpolarised', polarisation_value=0.0, **psd_kwargs):
    """
    Generates a colored noise pulse of a given length and spectral properties.

    Parameters:
    -----------
    N : int
        Number of samples.
    dt : float
        Time spacing.
    psd : Callable
        The power spectral density of the noise.
    central_frequency : float
        The frequency at which to center the pulse.
    rng_generator : numpy.random.Generator
        Random number generator for creating noise samples.
    polarisation : str, optional
        Pulse polarisation type: 'unpolarised', 'linear', or 'elliptical'. Default is 'unpolarised'.
    polarisation_value : float, optional
        Polarisation value for elliptical polarisation. Default is 0.0.
    **psd_kwargs : dict
        Any keyword arguments needed by the psd function.

    Returns:
    --------
    tuple of numpy.ndarray
        Two numpy arrays representing the real part of the inverse Fourier transform of the colored pulses.
    """

    # Generate random white noise and perform FFT
    h1_white = np.fft.fft(rng_generator.standard_normal(N))
    f = np.fft.fftfreq(N, d=dt)[:N]

    if(not isinstance(central_frequency, (float, int))):
        raise TypeError("central_frequency must be a number")
    if(central_frequency < 0):
        raise ValueError("central_frequency must be greater or equal to zero")


    # Calculate the power spectral density
    try:
        s1 = psd(f, central_frequency, **psd_kwargs)
    except KeyError:
        raise KeyError("PSDERROR: Required keywords were not passed through psd_kwargs")

    # Normalize the power spectral density
    s1 = s1 / np.sqrt(np.mean(s1**2))

    # Shape the noise by the power spectral density
    h1_shaped = h1_white * s1

    # Generate pulse based on the specified polarisation
    if polarisation == 'unpolarised':
        h2_white = np.fft.fft(rng_generator.standard_normal(N))[:N]
        h2_shaped = h2_white * s1
        return np.fft.ifft(h1_shaped)[:N].real, np.fft.ifft(h2_shaped)[:N].real

    elif polarisation == 'linear':
        h2_white = np.fft.fft(rng_generator.standard_normal(N))[:N]
        h2_shaped = h2_white * s1
        return np.fft.ifft(h1_shaped)[:N].real, np.zeros(N)

    elif polarisation == 'elliptical':
        h2_white = np.fft.fft(rng_generator.standard_normal(N))[:N]
        h2_shaped = 1.0j * h2_white * s1 * polarisation_value
        return np.fft.ifft(h1_shaped)[:N].real, np.fft.ifft(h2_shaped)[:N].real

    else:
        raise ValueError(f"{polarisation} is not a valid polarisation type.")


@singledispatch
def generate_mode(mode, *args, **kwargs):
    """
    Template function for the function that generates
    any given mode.
    """
    raise TypeError("Unsupported mode type")
    return 0


@generate_mode.register(abc.Callable)
def _(mode, time, dt, rng_generator, pulse_duration=0.05, polarisation='unpolarised',
      polarisation_value=0.0, mode_kwargs={}, psd=gauss_psd, psd_kwargs={'sigma': 10}):
    """
    Generates the signal component associated with a given mode.
    The signal is generated as a series of pulses of colored noise.

    Parameters:
    -----------
    mode : Callable
        Function describing the time evolution of the central frequency of the mode.
    time : list or numpy.ndarray
        Times at which to sample the signal.
    dt : float
        Sampling spacing in time.
    rng_generator : numpy.random.Generator
        Random number generator for creating noise samples.
    pulse_duration : float, optional
        Length of each of the individual pulses that make up the signal. Default is 0.05.
    polarisation : str, optional
        Pulse polarisation type: 'unpolarised', 'linear', or 'elliptical'. Default is 'unpolarised'.
    polarisation_value : float or numpy.ndarray, optional
        Polarisation value for elliptical polarisation. Default is 0.0.
    mode_kwargs : dict, optional
        Any keyword arguments needed by the mode function. Default is an empty dictionary.
    psd : Callable, optional
        PSD to use for the colored noise. Default is gauss_psd.
    psd_kwargs : dict, optional
        Any arguments needed by the psd function. Default is {'sigma': 10}.

    Returns:
    --------
    tuple of numpy.ndarray
        Two numpy arrays representing the x and p components of the signal.
    """

    # Validate and convert 'time' input
    if isinstance(time, (list, np.ndarray)):
        time = np.asarray(time)
        if len(time) <= 2:
            raise ValueError("time must have a length greater than 2")
    else:
        raise TypeError("time must be a list or a numpy array")

    # Validate and convert 'polarisation_value' input
    if isinstance(polarisation_value, (int, float)):
        epsilon = polarisation_value * np.ones(len(time))
    elif isinstance(polarisation_value, (list, np.ndarray)):
        if len(polarisation_value) == len(time):
            epsilon = np.asarray(polarisation_value)
        else:
            raise ValueError("Length of polarisation_value does not match the length of time")
    else:
        raise TypeError("polarisation_value must be a single number, list, or a numpy array")

    signal_length = len(time)
    N = int(pulse_duration / dt)

    # Calculate the central frequency using the provided mode function
    try:
        central_frequency = mode(**mode_kwargs)
    except KeyError:
        raise KeyError("MODEFUN ERROR: Required keywords were not passed through mode_kwargs")
    if len(time) != len(central_frequency):
        raise IndexError("Time array and mode array have different lengths")

    # Prepare zero-padded arrays for signal generation
    npad = int(pulse_duration / dt)
    signal_length = len(time)
    hp = np.zeros(signal_length + 2 * npad)
    hx = np.zeros(signal_length + 2 * npad)

    # Generate signal by iterating over the central_frequency array
    for i, f in enumerate(central_frequency):
        # Generate pulses of colored noise using the gen_pulse function
        hx_pulse, hp_pulse = gen_pulse(2 * npad, dt, psd, f, rng_generator,
                                       polarisation, epsilon[i], **psd_kwargs)
        # Generate random amplitude for scaling the pulses
        random_amplitude = (1 - 2 * rng_generator.random())

        # Add the generated pulses to the main signal arrays
        hx[i:i + 2 * npad] = hx[i:i + 2 * npad] + hx_pulse * random_amplitude
        hp[i:i + 2 * npad] = hp[i:i + 2 * npad] + hp_pulse * random_amplitude

    # Normalize and return the x and p components of the signal
    hxnorm = np.abs(hx[npad:-npad]).max()
    hpnorm = np.abs(hp[npad:-npad]).max()
    if(hpnorm == 0):
        hpnorm = 1
    if(hxnorm == 0):
        hxnorm = 1
        
    return hx[npad:-npad]/hxnorm, hp[npad:-npad]/hpnorm




@generate_mode.register(abc.Sequence)
@generate_mode.register(np.ndarray)
def _(mode, time, dt, rng_generator, pulse_duration=0.05, polarisation='unpolarised',
      polarisation_value=0.0, psd=gauss_psd, psd_kwargs={'sigma': 10},mode_kwargs={}):
    """
    Generates the signal component associated with a given mode.
    The signal is generated as a series of pulses of colored noise.

    Parameters:
    -----------
    mode : ndarray, list
        Array describing the time evolution of the central frequency of the mode.
    time : list or numpy.ndarray
        Times at which to sample the signal.
    dt : float
        Sampling spacing in time.
    rng_generator : numpy.random.Generator
        Random number generator for creating noise samples.
    pulse_duration : float, optional
        Length of each individual pulse that make up the signal. Default is 0.05.
    polarisation : str, optional
        Pulse polarisation type: 'unpolarised', 'linear', or 'elliptical'. Default is 'unpolarised'.
    polarisation_value : float or numpy.ndarray, optional
        Polarisation value for elliptical polarisation. Default is 0.0.
    psd : Callable, optional
        PSD to use for the colored noise. Default is gauss_psd.
    psd_kwargs : dict, optional
        Any arguments needed by the psd function. Default is {'sigma': 10}.

    Returns:
    --------
    tuple of numpy.ndarray
        Two numpy arrays representing the x and p components of the signal.
    """

    # Validate and convert 'time' input
    if isinstance(time, (list, np.ndarray)):
        time = np.asarray(time)
        if len(time) <= 2:
            raise ValueError("time must have a length greater than 2")
    else:
        raise TypeError("time must be a list or a numpy array")

    # Validate and convert 'polarisation_value' input
    if isinstance(polarisation_value, (int, float)):
        epsilon = polarisation_value * np.ones(len(time))
    elif isinstance(polarisation_value, (list, np.ndarray)):
        if len(polarisation_value) == len(time):
            epsilon = np.asarray(polarisation_value)
        else:
            raise ValueError("Length of polarisation_value does not match the length of time")
    else:
        raise TypeError("polarisation_value must be a single number, list, or a numpy array")

    signal_length = len(time)
    N = int(pulse_duration / dt)

    # Calculate the central frequency using the provided mode function

    central_frequency = np.asarray(mode)
    if(len(time) != len(central_frequency)):
        raise IndexError("Time array and mode array have different lengths")


    # Prepare zero-padded arrays for signal generation
    npad = int(pulse_duration / dt)
    signal_length = len(time)
    hp = np.zeros(signal_length + 2 * npad)
    hx = np.zeros(signal_length + 2 * npad)

    # Generate signal by iterating over the central_frequency array
    for i, f in enumerate(central_frequency):
        # Generate pulses of colored noise using the gen_pulse function
        hx_pulse, hp_pulse = gen_pulse(2 * npad, dt, psd, f, rng_generator,
                                       polarisation, epsilon[i], **psd_kwargs)
        # Generate random amplitude for scaling the pulses
        random_amplitude = (1 - 2 * rng_generator.random())

        # Add the generated pulses to the main signal arrays
        hx[i:i + 2 * npad] = hx[i:i + 2 * npad] + hx_pulse * random_amplitude
        hp[i:i + 2 * npad] = hp[i:i + 2 * npad] + hp_pulse * random_amplitude

    # Normalize and return the x and p components of the signal
    hxnorm = np.abs(hx[npad:-npad]).max()
    hpnorm = np.abs(hp[npad:-npad]).max()
    if(hpnorm == 0):
        hpnorm = 1
    if(hxnorm == 0):
        hxnorm = 1
        
    return hx[npad:-npad]/hxnorm, hp[npad:-npad]/hpnorm





def add_noise(signal_length, rng_generator, noise_level=0.1, signal_max=1.0):
    """
    Generates random noise for a signal of a given length. The strength of the
    noise is specified as a fraction of the maximum signal amplitude.

    Parameters:
    -----------
    signal_length : int
        The length of the signal, i.e., the number of noise samples needed.
    rng_generator : numpy random number generator
        The random number generator used to create the noise samples.
    noise_level : float, optional (default=0.1)
        The strength of the noise relative to the signal amplitude.
        The default value is set to 10% of the signal amplitude.
    signal_max : float, optional (default=1.0)
        The maximum value of the signal. The default value is 1.

    Returns:
    --------
    numpy.ndarray
        A 2D numpy array containing the generated noise for hx and hp.
    """
    noisehx = signal_max * noise_level * (-2 * rng_generator.random(signal_length) + 1)
    noisehp = signal_max * noise_level * (-2 * rng_generator.random(signal_length) + 1)
    return np.array([noisehx, noisehp])

def add_shaped_noise(signal_length,signal_dt, rng_generator,psd=cauchy_psd,
                     psd_kwargs={'x0' : -100,'gamma' : 800},central_frequency=500, noise_level=0.1, 
                     signal_max=1.0,polarisation='unpolarised', polarisation_value=0.0):
    """
    Generates shaped noise, based on specified PSD.

    Parameters:
    -----------
        signal_length (int): Length of the signal.
        signal_dt (float): Time interval (delta t) of the signal.
        rng_generator: Random number generator object.
        psd (function): Function that generates the Power Spectral Density. 
                        Default is 'colored_noise_psd'.
        psd_kwargs (dict): Keyword arguments for the PSD function. 
                           Defaults to {'beta': 0.5}.
        central_frequency (float): Central frequency for the PSD. Default is 100 Hz.
        noise_level (float): Overall level of the noise to be added. Default is 0.1.
        signal_max (float): Maximum value of the signal. Default is 1.0.
        polarisation (str): Polarisation type of the noise ('unpolarised' or other types). 
                            Default is 'unpolarised'.
        polarisation_value (float): Value associated with the polarisation. Default is 0.0.
    -----------
    Returns:
        ndarray: An array containing two channels (h1 and h2) of noise.

    """


    h1,h2 = gen_pulse(signal_length, signal_dt, psd, central_frequency, rng_generator,
              polarisation, polarisation_value,**psd_kwargs)
    
    return signal_max * noise_level*np.array([h1,h2])
