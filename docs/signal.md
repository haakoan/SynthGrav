---
layout: page
title:  Signal Class
description: Here we document the main signal class
---

# Signal Class
The Signal class represents a gravitational wave signal generated from a set of modes. For each mode and at each time step, the signal is created by randomly sampling from a window centered on the mode frequency at that time. Randomly sampled noise is then added to the signal. The relative strength of each mode can be adjusted by user-specified mode weights.

## Attributes
- `time`: numpy.array
  - An array containing the time coordinates. Assumed to be equidistant.

- `signal`: numpy.array
  - The gravitational wave signal sampled at the times in the `time` array.

- `modes_signal`: numpy.array
  - The gravitational wave signal sampled at the times in the `time` array, for each individual mode.

- `polarisation` : dict, optional
  - A dictionary specifying the polarisation state for each mode. 
    The keys should be the mode names, and the corresponding items should be a list containing two elements. 
    the first element represents the polarisation state
    indicating the type of polarisation, which can be one of ["unpolarised", "linear", "elliptical"].
    The second element should be a number giving the polarisation value, hc = hp*polarisation_value if elliptical.
    If a mode name is not included in this dictionary, it defaults to "unpolarised". 
    Default value is an empty dictionary {}.

- `modes`: dict
  - A dictionary listing the modes of the signal.

- `mode_weight`: dict
  - A dictionary containing weights corresponding to any one mode. The keys of `mode_weight` should correspond to the keys in `modes`. If a key in `modes` exists in `mode_weight`, then the mode's contribution to the total signal is taken to be `contribution_of_mode * mode_weight[mode]`.

- `signal_weight`: numpy.array
  - Time-dependent weights applied to the signal so that the total signal is given by `signal(t) * signal_weight`

- `rng_seed`: int
  - Seed used for the random number generator.

- `rng`: numpy.random.default_rng
  - An instance of the numpy random number generator, using the `rng_seed`.

- `dt`: float
  - Spacing between time steps.

## Methods

### `generate_signal()`
This function generates the gravitational wave signal based on the input provided by the user when the Signal was initiated. The user does not need to
provide any input since this has already been provided when initializing the Signal instance. For each mode in the signal, generate_mode() is called and a
mode weight is applied. The total signal is then constructed by summing up all the realizations of the modes and adding noise.

- Arguments: None
- Returns: None

While this function does not return anything, it modifies the Signal instance and sets `Signal.signal` and `Signal.modes_signal`

After the user has initialized a signal instance, the signal is generated as follows:
```python
signal.generate_signal()
```

### `save_all(filename)`
Saves various attributes of the Signal object to a tab-delimited text file. This function saves the time array, the signal components (`h1` and `h2`), individual modes, and noise data to a file. Each column in the output file corresponds to one of these attributes, and a header row is provided for easier identification.

- Arguments:
  - `filename` (`str`): The name of the output file where the data will be saved.    
- Returns: None

**Notes:**
The output file will be tab-delimited, and its first row will contain the names of the columns. The first column will be the time array, followed by the signal components (`h1` and `h2`), individual modes, and noise data. Each mode and polarization type will have its own column(s).

The header will also include a comment indicating the RNG seed used for generating the data, facilitating reproducibility.

### `save_signal(filename)`

#### Description
Saves the time array and the two polarization components (`h1` and `h2`) of the signal to a tab-delimited text file. The output file will consist of three columns: the time array, `h1`, and `h2`.

- Arguments:
  - `filename` (`str`): The name of the output file where the signal data will be saved.
- Returns: None
 
**Notes:**
The output file will be tab-delimited and the first row will serve as a header, specifying the names of the columns ('Time', 'h1', and 'h2'). 
A comment line will also be included in the header indicating the RNG seed used for generating the data, aiding in reproducibility.


### `__init__(self, time, modes, mode_kwargs={}, mode_weight = {}, polarisation=False, signal_weight = None, rng_seed = -1, noise_level=1.0)`

This method sets up the Signal instance based on the input provided by the user. The user can specify the desired modes either as a string (for a single mode), as a tuple of strings, or as a dictionary of user-defined modes. 

- Arguments:
  - `time`: numpy.array. Array with the time coordinates at which to sample the signal.
  - `modes`: str, tuple(str), dict. Modes to be included in the signal.
  - `mode_kwargs`: dict. A dict containing keyword arguments that are to be passed to the mode functions.
  - `mode_weight`: dict. A dict containing weights to adjust each individual mode's contribution to the total signal.
  - `signal_weight`: numpy.array. Array to weight the total signal, time dependent.
  - `rng_seed`: int. Seed used in random number generation.
  - `noise_level`: float. Level of noise to be added to the signal.
- Returns: None

---
