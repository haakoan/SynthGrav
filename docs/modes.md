---
layout: page
title:  Predefined modes
description: Here we document the signal modes
---
# Modes Sub-Library

The `modes` sub-library provides a collection of predefined modes and user-defined functions for generating gravitational wave signals. 
These modes can be used to construct custom gravitational wave signals using the `Signal` class.

## Predefined Modes

The predefined modes available in the `modes` sub-library are:

- `fmode`
- `p1mode`
- `p2mode`
- `p3mode`
- `g1mode`
- `g2mode`
- `g3mode`
- `gmode_default_func`

gmode_default_func is a test function and the other modes are based on 
Universal relations for gravitational-wave asteroseismology of proto-neutron stars, Alejandro Torres-Forné .et. al.
https://arxiv.org/abs/1902.10048

Each predefined mode is implemented as a function that generates a gravitational wave signal based on the given time array and mode-specific parameters.
The parameters of the modes are passed through the `Signal.mode_kwargs` dict. The dict is the same for every mode, but each mode is free to use the parts it needs
and nothing more. This is a practical way of coding the modes, but it might be sub-optimal if `Signal.mode_kwargs` grows large. In such cases, it is recommended to run
the signal generation using precomputed input arrays for the central frequency. For the code to work properly, `Signal.mode_kwargs` must contain elements that are 
numpy arrays and have the same length as the `Signal.time`.

Any number can be given to these modes, but for them to function as intended, the input must be reasonable and in accordance with the method described in 
Universal relations for gravitational-wave asteroseismology of proto-neutron stars.

### p1mode

`p1mode` generates a gravitational wave frequency based on the first pressure mode (p1-mode) of a neutron star. 
The function takes the mass (in solar masses) and radius (in km) as input parameters and returns a frequency array.

Parameters:
- msh: numpy.array - An array containing the mass values (in solar masses).
- rsh: numpy.array - An array containing the radius values (in km).

### p2mode

`p2mode` generates a gravitational wave frequency based on the second pressure mode (p2-mode) of a neutron star. 
The function takes the mass (in solar masses) and radius (in km) as input parameters and returns a frequency array.

Parameters:
- msh: numpy.array - An array containing the mass values (in solar masses).
- rsh: numpy.array - An array containing the radius values (in km).

### p3mode

`p3mode` generates a gravitational wave frequency based on the third pressure mode (p3-mode) of a neutron star. 
The function takes the mass (in solar masses) and radius (in km) as input parameters and returns a frequency array.

Parameters:
- msh: numpy.array - An array containing the mass values (in solar masses).
- rsh: numpy.array - An array containing the radius values (in km).

### g1mode

`g1mode` generates a gravitational wave frequency based on the first gravity mode (g1-mode) of a neutron star.
The function takes the mass (in solar masses) and radius (in km) as input parameters and returns a frequency array.

Parameters:
- mpns: numpy.array - An array containing the mass values (in solar masses).
- rpns: numpy.array - An array containing the radius values (in km).

### g2mode

`g2mode` generates a gravitational wave frequency based on the second gravity mode (g2-mode) of a neutron star. 
The function takes the mass (in solar masses) and radius (in km) as input parameters and returns a frequency array.

Parameters:
- mpns: numpy.array - An array containing the mass values (in solar masses).
- rpns: numpy.array - An array containing the radius values (in km).

### g3mode

`g3mode` generates a gravitational wave frequency based on the third gravity mode (g3-mode) of a neutron star. 
The function takes the mass (in solar masses), radius (in km), central pressure, and central density as input parameters and returns a frequency array.

Parameters:
- msh: numpy.array - An array containing the mass values (in solar masses).
- rsh: numpy.array - An array containing the radius values (in km).
- pC: numpy.array - An array containing the central pressure values.
- rhoC: numpy.array - An array containing the central density values.

### gmode_default_func

`gmode_default_func` is a simple toy mode function that generates a gravitational wave frequency based on time. 
It serves as a default function for testing purposes. The function takes time as an input parameter and returns a frequency array.

Parameters:
- time: numpy.array - An array containing the time values.


## User-Defined Modes

To use a custom mode function with the Signal class, you can pass it as a dictionary value to the modes parameter when initializing the Signal instance:
Users can create their own custom mode functions to generate gravitational wave signals. A user-defined mode function must have the following signature:

```python
def custom_mode_function(mode_kwargs):
    '''
    Arguments:
      mode_kwargs: dict. A dictionary containing any additional mode-specific parameters required by the custom mode function.
    Returns:
      central_frequency: numpy.array. An array with the same length as Signal.time describing the time evolution of the mode's central
      frequency
      '''
    return central_frequency
 ```
### Example
```python
import numpy as np
def custom_mode_function(**kwargs):
    t = kwargs['time']
    a = kwargs['a']
    b = kwargs['b']
    return a*t+b
  
my_modes = {'custom_mode': custom_mode_function}
my_mode_kwargs = {'time' : np.linspace(0,1,100), 'a' : 150, 'b' : 130 }
signal = Signal(time, my_modes, mode_kwargs=my_mode_kwargs)
``` 

