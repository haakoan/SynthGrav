---
title: 'Signal: Synthetic Gravitational-wave Signals for Core-collapse Supernovae'
tags:
  - Python
  - astronomy
  - supernova
  - gravitational waves
authors:
  - name: Haakon Andresen
    orcid: 0000-0002-4747-8453
    corresponding: true
    equal-contrib: true
    affiliation: "1"
  - name: Bella Finkel
    orcid: 0000-0002-9099-9713
    equal-contrib: true
    affiliation: "2"
affiliations:
 - name: The Oskar Klein Centre, Department of Astronomy, AlbaNova, SE-106 91 Stockholm, Sweden
   index: 1
 - name: University of Wisconsin--Madison, Department of Mathematics, Madison, WI 53706, USA
   index: 2
# - name: Independent Researcher, Country
#   index: 3
date:  August 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---
# Summary
Core-collapse supernovae (CCSNe) are promising gravitational wave (GW) sources that play an important role in the evolution of galaxies and the formation of heavy elements. Their GW emission contains information about the supernova explosion mechanism and the structure and rotation of the inner regions of the star. While the GW signal of a CCSN has yet to be detected, sophisticated simulations of CCSNe have provided insights into their explosive mechanisms and the GW emission associated with stellar collapse 
(see, for example, [@Andresen:2016pdt]) and have been used to relate the spectral properties of the signal to the
physical properties of the supernova core [@Torres-Forne:2018; @Morozova:2018; @Torres-Forne:2019b; @Sotani:21].

Despite these advancements, our understanding of CCSNe and their GW emissions is far from complete. 
The turbulent nature of multi-dimensional hydrodynamics makes supernovae challenging to model accurately.
State-of-the-art CCSNe simulations often take months to perform, utilizing world-leading supercomputers. 
As a result, relatively few predictions of the GW from CCSNe exist.
On the other hand, data analysis studies often implement techniques, such as machine learning, that rely on large sets of input data 
[@Astone:2018; @Mitra:2023; @Saiz-Perez:2022; Antelis:2022; @CasallasLagos:2023; @Iess:2020; @Chan:2020; @Lopez:2021]. The need for extensive data sets can not be met by resource-intensive CCSNe simulations.

The Signal software fills the gap between the needs of machine learning studies and the availability of GW signals from CCSNe simulations.
Signal is a flexible Python library for generating semi-analytic GW signals from CCSNe based on 
relations between the GW waveform and physical parameters of the CCSN [@Torres-Forne:2019b].
The software enables researchers to generate large sets of detailed signal predictions at a low computational cost, a crucial aspect for developing CCSNe-specific detection techniques and exploring the intricacies of the GWs produced by CCSNe.

# Statement of Need
Signal is designed to facilitate machine learning applications, parameter estimation studies, and other data analysis tasks, which
will help connect theoretical CCSN modelling and GW astronomy.
The Signal Python package offers an accessible solution to the data needs of modern data analysis techniques like machine learning, reducing the reliance on sparse existing predictions or the need to develop signal-generating simulation software.

# Generating a GW Signal 
In Signal, a GW signal is constructed from a weighted sum over a user-defined set of independent modes. 
The basic workflow is as follows.

1) Central Frequency Specification: The time evolution of the central frequency ($f_c$) needs to be 
specified for each mode. It can be given either as a function or as a numpy array. 

2) Mode Weights and Polarization Properties: Signal allows flexibility in determining the time-dependent 
mode weights and polarization properties. By default, weights are set uniformly, and signals are unpolarized.

3) Signal Weight Specification: The total weight of the signal determines the strength of the GW signal as a function of time. 

4) Signal Generation: The GW signal is generated using the parameters specified in the previous steps.

Except for step 4, each step allows for a great deal of customization and the software, consequently, offers the tools necessary to create
a wide variety of synthetic GW signals. SOFTWARENAME is designed for CCSNe, but can be used to generate
signals for other sources. 

# Mode Generation
Each mode is generated from a set of pulses of coloured noise, which allows for an accurate representation of the
stochastic signals observed from CCSNe. Individual pulses are created by colouring white noise such that the
power spectral density (PSD) of the pulse follows the distribution specified by the user.
A user-defined parameter determines the temporal width of the pulse, and the chosen PSD effectively sets the frequency bandwidth.
A mode is a sum of overlapping pulses in time and frequency space.

The following provides a step-by-step explanation of the pulse generation procedure.

1) **Generation of White Noise**: Initially, a sequence of random numbers is generated. This is denoted as $H_{\text{white}}$ and its Fourier Transform as $h_{\text{white}}$. 

2) **Calculation of PSD**: A PSD function $S(f,f_c)$ is defined. This PSD is responsible for colouring the noise and is centred around the central frequency of each mode. A Gaussian PSD is used by default.
The PSD is normalized as follows: $S_{\text{norm}}(f) = S(f,f_c) / \sqrt{\langle S(f,f_c))^2 \rangle}$. Here $\langle S(f,f_c)^2 \rangle$ denotes the mean of $S(f,f_c)^2$.

3) **Shaping the Noise**: The white noise generated in the first step is then weighted by PSD calculated in step 2. This is done by multiplying the normalized PSD with the Fourier transform of the white noise $h_{\text{white}}$ in the frequency domain. The shaped Fourier transform of the noise $h_{\text{shaped}}$ is then defined as $h_{\text{shaped}} = h_{\text{white}} \cdot S_{\text{norm}}(f)$.

4) **Inverse Fourier Transform**: The last step is to convert the shaped noise back into the time domain by performing an inverse Fourier transform on $h_{\text{shaped}}$. 

5) This results in the final coloured noise $H_{\text{shaped}}$, which is used to generate the GW signal.

# Acknowledgements

HA is supported by the Swedish Research Council (Project No. 2020-00452). BF is supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE-2137424.
Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors 
and do not necessarily reflect the views of the National Science Foundation. Support was also provided by the Graduate School and the Office of the Vice Chancellor for Research and Graduate Education at the University of Wisconsin-Madison with funding from the Wisconsin Alumni Research Foundation.

# References


