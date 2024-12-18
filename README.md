
# SynthGrav: A Python Package for Synthetic Gravitational-Wave Signal Generation

## Overview
SynthGrav is designed to facilitate machine learning applications, parameter estimation studies, and other data analysis tasks in the field of Core-Collapse Supernovae (CCSNe) and Gravitational Wave (GW) astronomy. 
The package aims to bridge the gap between theoretical CCSN modeling and GW astronomy by providing an accessible means of generating GW signals for data analysis. 
While SynthGrav is specifically designed for supernovae, its versatile framework makes it applicable to any burst source.


## Features

- Easy-to-use Python interface
- Designed for machine learning applications
- Ideal for parameter estimation studies
- Reduces reliance on sparse existing predictions
- Eliminates the need to develop signal-generating software

## Installation

Follow these steps to install SynthGrav:

1. Clone the repository:
    ```
    git clone https://github.com/haakoan/SynthGrav
    ```
2. Navigate to the directory where SynthGrav is located and import it in Python:
    ```python
    import synthgrav
    ```

## Getting Started

For a detailed guide on how to generate signals and use the software, refer to the Jupyter Notebook `Getting_started.ipynb` included in the repository.

## Documentation
Full documentation can be found at https://haakoan.github.io/GWSG/

## Unit Tests

We have implemented a suite of unit tests to ensure the reliability and robustness of SynthGrav. 
These tests cover key functionalities and aim to catch any regressions or errors that could affect the package's performance. 
Running the unit tests is highly recommended after making changes to the code. The tests can be found in the `unittest.py`.




## References

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For more information or to report issues, please contact haakon.andresen@astro.su.se.
