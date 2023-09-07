import unittest
import numpy as np
from scipy.fft import fft, fftfreq
from genpython import *
from datetime import datetime
from mode_list import *
from classSignal import *
from mode_list import *

"""
This unit test suite covers the core functions of the project. 
In each test, we validate the proper execution of the functions and 
check if input arguments are processed correctly.

The tests implemented include:

1. Input validation for main class
2. Test of different mode generation fucntions
3. Test for PSD functions
4. Test input validation for gen_pulse
5. Test for input validation for generate_mode

However, these tests do not cover all possible edge cases and scenarios. Therefore, users are strongly 
encouraged to validate their own inputs and outputs when using these functions. If anything looks 
unexpected or inconsistent, please verify your parameters and consult the function documentation 
to ensure correct usage. 

The purpose of these tests is to provide a safety net for common usage, but the user has 
the final responsibility to verify that the results are consistent with their specific use case.
"""



# Simple mode function for testing
def simple_mode(**mode_kwargs):
    freq = mode_kwargs.get('freq', 50)
    N = mode_kwargs['N']
    return np.ones(N) * freq

def custom_mode(**mode_kwargs):
    t = mode_kwargs['time']
    return np.sin(t)

class TestSignalClass(unittest.TestCase):
    def test_generate_signal_length(self):
        # Test parameters
        time = np.linspace(0, 1, 100)
        modekw = {'freq': 50, 'N': len(time)}
        mode = {'simple_mode': simple_mode}
        pol = {'simple_mode': ["unpolarised",0,0]}
        gws = Signal(time=time,modes=mode,mode_kwargs=modekw,polarisation=pol)
        # Generate signal using the provided function
        gws.generate_signal()
        hx, hp = gws.signal[0],gws.signal[1]


        # Check if the generated signal length matches the input time array length
        self.assertEqual(len(hx), len(time), msg="Length of hx does not match length of time array")
        self.assertEqual(len(hp), len(time), msg="Length of hp does not match length of time array")

    def test_spectral_content(self):
        # Test parameters
        from scipy import signal
        from scipy.signal import filter_design as fd
        time = np.linspace(0, 2, 10800)
        modekw = {'freq': 250, 'N': len(time)}
        mode = {'simple_mode': simple_mode}
        pol = {'simple_mode': ["unpolarised",0,0]}
        gws = Signal(time=time,modes=mode,mode_kwargs=modekw,polarisation=pol,pulse_duration=0.05,noise_level=0.1)
        # Generate signal using the provided function
        
        gws.generate_signal()
        hx, hp = gws.signal[0],gws.signal[1]

        #When dealing with FFTs it is always an issue to get the exact frequency. This test is
        #somewhat sensetive to exactly how it is setup, so be careful with making changes.
        expected_peak_freq = 250.0

        # Calculate the power spectral density (PSD) of the generated signal
        dt = time[1]-time[0]
        freqs = fftfreq(len(time), dt)

        fnyq = max(freqs)
        bl, al = signal.butter(2, 2000.0/fnyq,'low')
        bh, ah = signal.butter(2, 25.0/fnyq,'high')
   
        window = signal.blackman(len(time))

        hx = signal.filtfilt(bh, ah, hx)
        hx = signal.filtfilt(bl, al, hx)*window

        hp = signal.filtfilt(bh, ah, hp)
        hp = signal.filtfilt(bl, al, hp)*window

        hx_fft = np.abs(np.fft.fft(hx))
        hp_fft = np.abs(np.fft.fft(hp))

        # Find the peak frequency in the PSD
        peak_freq_hx = freqs[np.argmax(hx_fft)]
        peak_freq_hp = freqs[np.argmax(hp_fft)]

        # Check if the peak frequency matches the expected frequency
        self.assertTrue(peak_freq_hx-1.0/dt <= modekw['freq'] <= peak_freq_hx+1.0/dt,msg="Peak frequency in hx does not match expected central frequency")
        self.assertTrue(peak_freq_hp-1.0/dt <= modekw['freq']  <= peak_freq_hp+1.0/dt,msg="Peak frequency in hp does not match expected central frequency")
        
class TestGenerateModeFunction(unittest.TestCase):
    def test_invalid_time_input(self):
        # Test parameters
        invalid_time_inputs = [
            42,
            "invalid_time",
            np.array([1, 1])
        ]
        dt = 0.01
        rng_generator = np.random.default_rng(seed=42)
        pulse_duration = 0.05
        mode_kwargs = {'freq': 50}

        # Test invalid time inputs
        for invalid_time in invalid_time_inputs:
            with self.assertRaises((TypeError, ValueError),
                                   msg=f"Function did not raise an error for invalid time input: {invalid_time}"):
                generate_mode(simple_mode, invalid_time, dt, rng_generator, pulse_duration=pulse_duration,
                  polarisation='unpolarised', mode_kwargs=mode_kwargs)

    def test_invalid_polarisation_value_input(self):
        # Test parameters
        time = np.linspace(0, 1, 100)
        dt = time[1] - time[0]
        rng_generator = np.random.default_rng(seed=42)
        pulse_duration = 0.05
        mode_kwargs = {'freq': 50}
        invalid_polarisation_value_inputs = [
            "invalid_polarisation_value",
            np.array([1, 1, 1])
        ]

        # Test invalid polarisation_value inputs
        for invalid_polarisation_value in invalid_polarisation_value_inputs:
            with self.assertRaises((TypeError, ValueError),
                                   msg=f"Function did not raise an error for invalid polarisation_value input: {invalid_polarisation_value}"):
                generate_mode(simple_mode, time, dt, rng_generator, pulse_duration=pulse_duration,
                  polarisation='unpolarised', polarisation_value=invalid_polarisation_value, mode_kwargs=mode_kwargs)




class TestGenPulseFunction(unittest.TestCase):
    def test_pulse_length(self):
        # Test parameters
        N = 100
        dt = 0.01
        central_frequency = 50
        rng_generator = np.random.default_rng(seed=42)

        # Generate pulse using the provided function
        hx, hp = gen_pulse(N, dt, gauss_psd, central_frequency, rng_generator,sigma = 10)

        # Check if the generated pulse length matches the input N
        self.assertEqual(len(hx), N, msg="Length of hx does not match N")
        self.assertEqual(len(hp), N, msg="Length of hp does not match N")

    def test_spectral_content(self):
        # Test parameters
        N = 500
        dt = 0.01
        central_frequency = 50
        rng_generator = np.random.default_rng(seed=42)

        # Generate pulse using the provided function
        hx, hp = gen_pulse(N, dt, gauss_psd, central_frequency, rng_generator,sigma = 10)

        # Calculate the power spectral density (PSD) of the generated pulse
        hx_fft = np.abs(fft(hx))
        hp_fft = np.abs(fft(hp))
        freqs = fftfreq(N, dt)

        # Find the peak frequency in the PSD
        peak_freq_hx = freqs[np.argmax(hx_fft)]
        peak_freq_hp = freqs[np.argmax(hp_fft)]

        # Check if the peak frequency matches the expected central frequency
        self.assertTrue(peak_freq_hx-1.0/dt <= central_frequency <= peak_freq_hx+1.0/dt,msg="Peak frequency in hx does not match expected central frequency")
        self.assertTrue(peak_freq_hp-1.0/dt <= central_frequency <= peak_freq_hp+1.0/dt,msg="Peak frequency in hp does not match expected central frequency")


    def test_invalid_psd_kwargs(self):
        # Test parameters
        N = 100
        dt = 0.01
        central_frequency = 50
        rng_generator = np.random.default_rng(seed=42)
        invalid_psd_kwargs = {'invalid_key': 42}

        # Test invalid psd_kwargs input
        with self.assertRaises(KeyError, msg="Function did not raise an error for invalid psd_kwargs input"):
            gen_pulse(N, dt, gauss_psd, central_frequency, rng_generator, psd_kwargs=invalid_psd_kwargs)


    def test_invalid_N(self):
        # Test parameters
        N = -10
        dt = 0.01
        central_frequency = 50
        rng_generator = np.random.default_rng(seed=42)

        # Test invalid N input
        with self.assertRaises(ValueError, msg="Function did not raise an error for invalid N input"):
            gen_pulse(N, dt, gauss_psd, central_frequency, rng_generator,sigma = 10)

    def test_invalid_central_frequency(self):
        # Test parameters
        N = 100
        dt = 0.01
        central_frequency = -50
        rng_generator = np.random.default_rng(seed=42)

        # Test invalid central_frequency input
        with self.assertRaises(ValueError, msg="Function did not raise an error for invalid central_frequency input"):
            gen_pulse(N, dt, gauss_psd, central_frequency, rng_generator,sigma = 10)

    def test_invalid_polarisation(self):
        # Test parameters
        N = 100
        dt = 0.01
        central_frequency = 50
        rng_generator = np.random.default_rng(seed=42)
        invalid_polarisation = "invalid_polarisation"

        # Test invalid polarisation input
        with self.assertRaises(ValueError, msg="Function did not raise an error for invalid polarisation input"):
            gen_pulse(N, dt, gauss_psd, central_frequency, rng_generator, polarisation=invalid_polarisation,sigma=10)

    def test_invalid_rng_generator(self):
        # Test parameters
        N = 100
        dt = 0.01
        central_frequency = 50
        invalid_rng_generator = "invalid_rng_generator"

        # Test invalid rng_generator input
        with self.assertRaises(AttributeError, msg="Function did not raise an error for invalid rng_generator input"):
            gen_pulse(N, dt, gauss_psd, central_frequency, invalid_rng_generator,sigma=10)



class TestAddNoiseFunction(unittest.TestCase):
    def test_add_noise_output_shape(self):
        signal_length = 100
        rng_generator = np.random.default_rng(42)
        noise_level = 0.1
        signal_max = 1.0

        noise = add_noise(signal_length, rng_generator, noise_level, signal_max)

        self.assertEqual(noise.shape, (2, signal_length),
                         "Output shape is incorrect")

    def test_add_noise_amplitude(self):
        signal_length = 1000
        rng_generator = np.random.default_rng(42)
        noise_level = 0.1
        signal_max = 1.0

        noise = add_noise(signal_length, rng_generator, noise_level, signal_max)

        # Check if the maximum amplitude of the noise is within the expected range
        self.assertTrue(np.max(np.abs(noise)) <= signal_max * noise_level,
                        "Maximum noise amplitude is too high")

    def test_add_noise_negative_signal_length(self):
        signal_length = -100
        rng_generator = np.random.default_rng(42)
        noise_level = 0.1
        signal_max = 1.0

        with self.assertRaises(ValueError, msg="Function did not raise an error for negative signal_length"):
            add_noise(signal_length, rng_generator, noise_level, signal_max)

    def test_add_noise_invalid_rng_generator(self):
        signal_length = 100
        invalid_rng_generator = "invalid_rng_generator"
        noise_level = 0.1
        signal_max = 1.0

        with self.assertRaises(AttributeError, msg="Function did not raise an error for invalid rng_generator input"):
            add_noise(signal_length, invalid_rng_generator, noise_level, signal_max)


class TestSignalGeneration(unittest.TestCase):

    def setUp(self):
        self.time = np.linspace(0, 0.4, 2000)
        self.N = len(self.time)
        self.modekw = {"rsh" : 150*np.ones(self.N),
                       "rpns" : 50*np.ones(self.N),
                        "mpns" : 2.1*np.ones(self.N),
                        "msh" : 2*np.ones(self.N),
                        "time" : self.time}

    def test_generate_signal_single_mode(self):
        pol = {'fmode' : ["unpolarised",0,0]}
        signal = Signal(self.time, modes='fmode',mode_kwargs=self.modekw,polarisation=pol)
        signal.generate_signal()
        self.assertIsNotNone(signal.signal)
        self.assertIsNotNone(signal.modes_signal)
        self.assertEqual(signal.signal.shape, (2, len(self.time)))

    def test_generate_signal_multiple_modes(self):
        pol = {'fmode' : ["unpolarised",0,0],
               'g1mode' : ["unpolarised",0,0]}
        signal = Signal(self.time, modes=('fmode', 'g1mode'),mode_kwargs=self.modekw,polarisation=pol)
        signal.generate_signal()
        self.assertIsNotNone(signal.signal)
        self.assertIsNotNone(signal.modes_signal)
        self.assertIn('fmode', signal.modes_signal)
        self.assertIn('g1mode', signal.modes_signal)
        self.assertEqual(signal.signal.shape, (2, len(self.time)))

    def test_generate_signal_custom_mode(self):
        pol = {'custom' : ["unpolarised",0,0]}
        signal = Signal(self.time, modes={'custom': custom_mode},mode_kwargs=self.modekw,polarisation=pol)
        signal.generate_signal()
        self.assertIsNotNone(signal.signal)
        self.assertIsNotNone(signal.modes_signal)
        self.assertIn('custom', signal.modes_signal)
        self.assertEqual(signal.signal.shape, (2, len(self.time)))

    def test_generate_signal_with_noise(self):
        pol = {'fmode' : ["unpolarised",0,0]}
        signal = Signal(self.time, 'fmode', noise_level=0.5,mode_kwargs=self.modekw,polarisation=pol)
        signal.generate_signal()
        self.assertIsNotNone(signal.noise)
        self.assertEqual(signal.noise.shape, (2, len(self.time)))

    def test_generate_signal_with_mode_weight(self):
        pol = {'fmode' : ["unpolarised",0,0]}
        signal = Signal(self.time, 'fmode', mode_weight={'fmode': 0.5},mode_kwargs=self.modekw,polarisation=pol)
        signal.generate_signal()
        self.assertIn('fmode', signal.mode_weight)
        self.assertEqual(signal.mode_weight['fmode'], 0.5)

    def test_generate_signal_normalized(self):
        pol = {'fmode' : ["unpolarised",0,0]}
        signal = Signal(self.time, 'fmode',mode_kwargs=self.modekw,polarisation=pol)
        signal.generate_signal()
        max_amplitude = np.abs(signal.signal).max()
        self.assertAlmostEqual(max_amplitude, 1.0, places=5)


class TestPSDFunctions(unittest.TestCase):

    def setUp(self):
        self.time = np.linspace(0, 0.4, 2000)
        self.N = len(self.time)
        self.frequencies = np.linspace(0, 100, num=self.N)
        self.central_frequency = 50
        self.kwargs = {'sigma': 10, 'gamma': 0.5, 'constant_value': 1, 
                       'delta_f': 20, 'mu' : 0.5, 'x0': 5.0, 's' : 10}
        
        self.modekw = {"rsh" : 150*np.ones(self.N),
                       "rpns" : 50*np.ones(self.N),
                        "mpns" : 2.1*np.ones(self.N),
                        "msh" : 2*np.ones(self.N),
                        "time" : self.time}


    def test_gauss_psd(self):
        psd_values = gauss_psd(self.frequencies, self.central_frequency, **self.kwargs)
        # Test that the maximum value is at the central frequency
        max_index = np.argmax(psd_values)
        self.assertAlmostEqual(self.frequencies[max_index], self.central_frequency, delta=self.central_frequency*0.10)

    def test_logistic_psd(self):
        psd_values = logistic_psd(self.frequencies, self.central_frequency, **self.kwargs)
        # Test that the maximum value is at the central frequency
        max_index = np.argmax(psd_values)
        self.assertAlmostEqual(self.frequencies[max_index], self.central_frequency, delta=self.central_frequency*0.10)

    def test_cauchy_psd(self):
        psd_values = cauchy_psd(self.frequencies, self.central_frequency, **self.kwargs)
        # Test that the maximum value is at the central frequency
        max_index = np.argmax(psd_values)
        self.assertAlmostEqual(self.frequencies[max_index], self.central_frequency, delta=self.central_frequency*0.10)

    def test_constant_psd(self):
        psd_values = constant_psd(self.frequencies, self.central_frequency, **self.kwargs)
        # Test that PSD value is 1 within the band and 0 outside
        in_band_indices = np.where(np.abs(self.frequencies - self.central_frequency) <= self.kwargs['delta_f'] / 2)
        out_band_indices = np.where(np.abs(self.frequencies - self.central_frequency) > self.kwargs['delta_f'] / 2)
        self.assertTrue(np.all(psd_values[in_band_indices] == self.kwargs['constant_value']))
        self.assertTrue(np.all(psd_values[out_band_indices] == 0))

    def test_constant_psd(self):
        pol = {'fmode' : ["unpolarised",0,0]}
        sig = Signal(self.time, 'fmode', noise_level=0.5,mode_kwargs=self.modekw,polarisation=pol,
                     psd=constant_psd,psd_kwargs=self.kwargs)
        sig.generate_signal()
        self.assertTrue(np.all(np.isfinite(sig.signal[0])), "Non-finite values found in signal generated with constant PSD.")
        self.assertTrue(np.all(np.isfinite(sig.signal[1])), "Non-finite values found in signal generated with constant PSD.")
        self.assertEqual(len(sig.signal[0]), len(self.time), "Output signal length doesn't match input time array for constant PSD.")
        self.assertTrue(np.max(np.abs(sig.signal[0])) <= 1, "Max absolute value greater than 1 for constant PSD.")

    def test_logistic_psd(self):
        pol = {'fmode' : ["unpolarised",0,0]}
        sig = Signal(self.time, 'fmode', noise_level=0.5,mode_kwargs=self.modekw,polarisation=pol,
                     psd=logistic_psd,psd_kwargs=self.kwargs)
        sig.generate_signal()
        self.assertTrue(np.all(np.isfinite(sig.signal[0])), "Non-finite values found in signal generated with logistic PSD.")
        self.assertTrue(np.all(np.isfinite(sig.signal[1])), "Non-finite values found in signal generated with logistic PSD.")
        self.assertEqual(len(sig.signal[0]), len(self.time), "Output signal length doesn't match input time array for logistic PSD.")
        self.assertTrue(np.max(np.abs(sig.signal[0])) <= 1, "Max absolute value greater than 1 for logistic PSD.")

    def test_cauchy_psd(self):
        pol = {'fmode' : ["unpolarised",0,0]}
        sig = Signal(self.time, 'fmode', noise_level=0.5,mode_kwargs=self.modekw,polarisation=pol,
                     psd=cauchy_psd,psd_kwargs=self.kwargs)
        sig.generate_signal()
        self.assertTrue(np.all(np.isfinite(sig.signal[0])), "Non-finite values found in signal generated with Cauchy PSD.")
        self.assertTrue(np.all(np.isfinite(sig.signal[1])), "Non-finite values found in signal generated with Cauchy PSD.")
        self.assertEqual(len(sig.signal[0]), len(self.time), "Output signal length doesn't match input time array for Cauchy PSD.")
        self.assertTrue(np.max(np.abs(sig.signal[0])) <= 1, "Max absolute value greater than 1 for Cauchy PSD.")


    def test_gauss_psd(self):
        pol = {'fmode' : ["unpolarised",0,0]}
        sig = Signal(self.time, 'fmode', noise_level=0.5,mode_kwargs=self.modekw,polarisation=pol,
                     psd=gauss_psd,psd_kwargs=self.kwargs)
        sig.generate_signal()
        self.assertTrue(np.all(np.isfinite(sig.signal[0])), "Non-finite values found in signal generated with Gauss PSD.")
        self.assertTrue(np.all(np.isfinite(sig.signal[1])), "Non-finite values found in signal generated with Gauss PSD.")
        self.assertEqual(len(sig.signal[0]), len(self.time), "Output signal length doesn't match input time array for Gauss PSD.")
        self.assertTrue(np.max(np.abs(sig.signal[0])) <= 1, "Max absolute value greater than 1 for Gauss PSD.")

if __name__ == "__main__":
    unittest.main()
