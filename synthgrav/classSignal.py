import numpy as np
from .genpython import generate_mode, add_noise,gauss_psd
from datetime import datetime
from .mode_list import *
# import gensig


class Signal:
    '''
    A class to represent a gravitational wave signal. A signal is built from a set of modes.
    For each mode and at each time step, the signal is generated by randomly sampling from a window centered on the
    mode frequency at the time, and then adding randomly sampled noise to the signal.
    The relative strength of each mode can be adjusted by user-specified mode weights.

    Attributes
      ----------
      time : numpy.array
          Array containing the time coordinates. Assumed to be equidistant.

      signal : numpy.array
          The gravitational wave signal sampled at the times in the time array.

      modes : dict
          A dictionary listing the modes of the signal.

      mode_weight : dict
          A dictionary containing weights corresponding to any one mode.
          The keys of mode_weight should correspond to the keys in modes.
          If a key in modes exists in mode_weight then the mode's contribution to the
          total signal is taken to be contribution_of_mode * mode_weight[mode].
      
      polarisation : dict 
          A dictionary where the mode names are the keys. 
          Each corresponding value is a 2-element array. The first element of the array represents the polarization state of that mode, 
          and the second element is an array representing the polarization values across the time domain for that mode.

    noise : np.array 
        A 2D NumPy array where each row represents one of the two gravitational-wave polarizations (h1 and h2). 
        The array has the same length as the time domain, and contains the colored noise added to each corresponding gravitational-wave polarization.


      signal_weight : numpy.array
          Time dependent weights applied to the signal, so that the
          total signal is given by signal(t) * signal_weight(t)

      rng_seed : int
          Seed used for the random number generator.

      rng : numpy.random.default_rng
          An instance of the numpy random number generator, using the rng_seed.

      dt : float
          Spacing between time steps.

    Methods
      -------
      generate_signal():
          Generates the gravitational wave signal based on the input given by the user when Signal was initiated.
      '''

    def __init__(self, time, modes, mode_kwargs={}, mode_weight={}, polarisation={},
                 signal_weight=None, rng_seed=-1, noise_level=1.0,
                 pulse_duration=0.05,psd=gauss_psd,psd_kwargs={'sigma':10}) -> None:
        '''
        Initializes the Signal instance based on the input provided by the user.

        Parameters
          ----------
          time : numpy.array
              Array with the time coordinates at which to sample the signal.

          modes : str, tuple(str), dict
              Modes to be included in the signal. If str or a tuple of str, then the value(s) must
              correspond to the modes implemented in the modes sub-library. The currently predefined modes are
              fmode, p1mode, p2mode, p3mode, g1mode, g2mode, g3mode, and gmode_default_func. If one of these modes are
              specified, then mode_kwargs must contain arrays of length len(time), specifying the necessary input parameters.

          mode_kwargs : dict
              A dict containing keyword arguments that are to be passed to any of the mode functions. The dict will be passed to
              each mode function, but not every mode function has to use every entry in the dict.

          mode_weight : dict
              A dict containing weights to adjust each individual mode's contribution to the total signal. The dict must have
              keys corresponding to the name given to modes in the modes dict, and each value must be a scalar or array of length len(time).

          signal_weight : numpy.array, optional
              Array to weight the total signal, time dependent. Such that signal = signal(t) * signal_weight(t)

          rng_seed : int, optional
              Seed used in random number generation. Default value -1; in this case, a datetime timestamp is chosen as the seed.

          noise_level : float, optional
              The level of noise to be added to the signal. Default is 1.0.

          polarisation : dict, optional
              A dictionary specifying the polarisation state for each mode. 
              The keys should be the mode names, and the corresponding items should be a list containing two elements. 
              The first element represents the polarisation state
              indicating the type of polarisation, which can be one of ["unpolarised", "linear", "elliptical"].
              The second element should be a number giving the polarisation value, hc = hp*polarisation_value if elliptical.
              If a mode name is not included in this dictionary, it defaults to "unpolarised". 
              Default value is an empty dictionary {}.
                    
        pulse_duration : float or dict, optional
              Determines the length of each pulse that make up the modes. Must either be a single float or a dict
              with one number per mode, keys must be the same as in modes.
        '''

        self.modes = {}
        self.rng_seed = rng_seed
        self.mode_weight = mode_weight
        self.mode_kwargs = mode_kwargs
        self.polarisation = {}
        self.psd = psd
        self.psd_kwargs = psd_kwargs

        #Validate time input
        if(isinstance(time,(list,tuple,np.ndarray))):
            self.time = np.asarray(time)
        else:
            raise ValueError("time must be array like")
        
        self.dt = self.time[1]-self.time[0]

        if(not self.dt > 0):
            raise ValueError("time must monotonic")
        
        #Set up signal container
        self.signal = np.zeros([2, len(self.time)])
        
        #Validate noise input
        if(isinstance(noise_level,(int,float))):
            self.noise_level = noise_level
        else:
            raise ValueError("noise_level must be int or float")
        
        #Validate mode input and set up modes 
        if(isinstance(modes, str)):
            try:
                self.modes[modes] = builtin_modes[modes]
            except:
                print(f"WARNING: {modes} is not found in the list of avilaible modes, initilaizing to default modes.")
        elif(isinstance(modes, tuple)):
            for mode in modes:
                try:
                    self.modes[mode] = builtin_modes[mode]
                except:
                    print(f"WARNING: {mode} is not found in the list of avilaible modes, skipping {mode}.")
        elif(isinstance(modes, dict)):
            self.modes = modes
        else:
            raise ValueError(
                "Modes must be gvien as a specific mode(str), a list of modes (str tuple), or a set of well defined user specified modes (dict).")
        #Set to default mode if empty input
        if(self.modes == {}):
            self.modes['default'] = gmode_default_func
            self.mode_kwargs = {'time': self.time}

        #A default weight that works well for SN
        if(not signal_weight):
            self.signal_weight = sigmoid((self.time-0.03)/self.time.max() * 10)**5
        else:
            if(isinstance(signal_weight,(list,tuple,np.ndarray))):
                self.signal_weight = np.asarray(signal_weight)
                if(len(self.signal_weight) != len(self.time)):
                    raise ValueError("signal_weight must be same length as time") 
            else:
                raise ValueError("signal_weight must be array like")


        #Setup RNG
        if(self.rng_seed == -1):
            self.rng_seed = int(datetime.now().timestamp())
            self.rng = np.random.default_rng(self.rng_seed)
        else:
            self.rng = np.random.default_rng(self.rng_seed)

        #Validate polarisation
        if (isinstance(polarisation, dict)):
            self.polarisation = polarisation
        else:
            raise ValueError(
                "Polarisation must be a dict where each requested mode is a key "
                "and the corresponding element is a list tuple (containing "
                "polarisation state [unpolarised,linear,elliptical] and polarisation value")

        for key in self.polarisation.keys():
            if(key not in self.modes):
                print(f"WARNING: Requested polarisation for mode {key} which does not exist, skipping {key}.")
                self.polarisation[key][0] = 'unpolarised'
                self.polarisation[key][1] = 0
            else:
                valid_polarisation_types = ["unpolarised", "linear", "elliptical"]
                if not isinstance(self.polarisation[key][0], str):
                    raise ValueError(f"The polarisation type for key '{key}' must be a string.")

                if self.polarisation[key][0].lower() not in ("unpolarised", "linear", "elliptical"):
                    raise ValueError(f"The polarisation type for key '{key}' must be one of unpolarised, linear, elliptical.")
                self.polarisation[key][0] = self.polarisation[key][0].lower()
                self.polarisation[key][1] = self.polarisation[key][1] * np.ones(len(self.time))

        
        if (isinstance(pulse_duration, dict)):
            self.pulse_duration = pulse_duration
        elif(isinstance(pulse_duration, float)):
            self.pulse_duration =  {}
            for key in self.modes.keys():
                self.pulse_duration[key] = pulse_duration
        else:
            raise ValueError("Invalid pulse duration, must be float or dict")

          


    def generate_signal(self) -> None:
        '''
        Generates the gravitational wave signal according to the user specifications provided during the Signal instance initialization.
        The signal will only contain the signal parameters until this function is called. Users do not need to provide any input,
        as it has already been provided when initializing the Signal instance.

        Returns
        -------
        None
            The method updates the signal attribute of the Signal instance in place and does not return any value.
        '''
        
        self.modes_signal = {}
        #Generate each mode
        for key in self.modes.keys():
            self.modes_signal[key] = generate_mode(self.modes[key], self.time, self.dt, self.rng, mode_kwargs=self.mode_kwargs,
                                                   polarisation=self.polarisation[key][0],
                                                   polarisation_value=self.polarisation[key][1],pulse_duration=self.pulse_duration[key],
                                                   psd=self.psd,psd_kwargs=self.psd_kwargs)
        #Add noise
        if(key in self.mode_weight):
            self.modes_signal[key] = np.array(self.modes_signal[key])*self.mode_weight[key]
        self.noise = add_noise(len(self.time), self.rng,
                               noise_level=self.noise_level)
        #Construct signal as a weighted sum of modes
        for key in self.modes_signal.keys():
            self.signal += np.array(self.modes_signal[key])
        self.signal += self.noise
        self.signal = self.signal*self.signal_weight
        self.signal = self.signal/np.abs(self.signal).max()


    def save_signal(self, filename):
        """
        Save time, and the signal data to file.

        Args:
            filename (str): The name of the file.
        """

        header = f'#Generated with seed {self.rng_seed}\nTime\th1\th2'
        data = np.column_stack((self.time, self.signal[0,:], self.signal[1,:]))
        np.savetxt(filename, data, delimiter='\t', header=header, comments='', fmt='%1.6e')

    def save_all(self, filename):
        """
        Save the attributes of the Signal object to a tab-delimited text file.

        This function saves the time array, the signal (h1 and h2 components), 
        individual modes, and noise data to a file. Each column in the output file
        corresponds to one of these attributes, and a header is provided for easier
        identification.

        Args:
            filename (str): The name of the output file where the data will be saved.

        Returns:
            None

        Notes:
            The output file will be tab-delimited and its first row will contain 
            the names of the columns. The first column will be the time array,
            followed by the signal components (h1 and h2), individual modes, and
            noise data. Each mode and polarization type will have its own column(s).

            The header will also include a comment indicating the RNG seed used for 
            generating the data, facilitating reproducibility.
        """

        mode_names = []
        polarisation_names = []
        
        # Stack arrays together
        out_data = np.column_stack((self.time,self.signal[0,:]))
        out_data = np.column_stack((out_data,self.signal[1,:]))
        for key in self.modes_signal.keys():
            out_data = np.column_stack((out_data, self.modes_signal[key][0], self.modes_signal[key][1]))
            mode_names.append(f'{key}_h1')
            mode_names.append(f'{key}_h2')
        for key in self.polarisation.keys():
            out_data = np.column_stack((out_data, self.polarisation[key][1]))
            polarisation_names.append(f'pol_{key} ({self.polarisation[key][0]})')
        out_data = np.column_stack((out_data, self.noise[0,:],self.noise[1,:]))

        column_names = ["time", "h1", "h2"] + list(mode_names) + list(polarisation_names) + ['noise h1','noise h2']
        header = f'# Generated with seed {self.rng_seed}\n #'
        for col in column_names:
            header+=f'\t{col}'

        np.savetxt(filename, out_data, delimiter='\t', header=header, comments='', fmt='%1.6e')



        


# Helper functions, this is need for the default signal weight.
def sigmoid(t):
  """
  Computes the sigmoid function for the given input.

  The sigmoid function is a smooth, S-shaped curve that maps any input value to a value between 0 and 1.

  Parameters
  ----------
  t : float or numpy.array
      Input value(s) for which the sigmoid function should be computed. Can be a single value or an array of values.

  Returns
  -------
  float or numpy.array
    The sigmoid function value(s) corresponding to the input value(s).
  """
  return 1/(1 + np.exp(-t))

