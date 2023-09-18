from setuptools import setup
setup(
    name='SynthGrav',
    version='1.0',
    packages=['synthgrav'],
   install_requires=[
        'numpy>=1.22',  
        'scipy>=1.7', 
    ]
    url='https://github.com/yourusername/SynthGrav',
    license='MIT',
    author='Haakon Andresen, Bella Finkel',
    author_email='haakon.andresen@astro.su.se',
    description='A Python package for Core-Collapse Supernovae and Gravitational Wave Astronomy',
    classifiers=[
        'Development Status :: 5 - Production/Stable', 
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
    ],
)
