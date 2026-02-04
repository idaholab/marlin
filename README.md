 <img src="https://github.com/idaholab/marlin/blob/devel/doc/content/marlin.png?raw=true" width="80%" alt="Marlin Logo">

**Marlin** is a device independent Fourier spectral solver application based on the [MOOSE Finite Element Framework](http://mooseframework.org). Marlin supports[^1] CPU, CUDA, and MPS[^2] with automatic detection of supported device precision.
[<img align="right" src="https://civet.inl.gov/idaholab/marlin/main/branch_status.svg/">](https://civet.inl.gov/repo/idaholab/marlin/)

[^1]: more compute device types might be supported, but have not been tested.
[^2]: torch MPS supports only single precision calculations!

> Note: Marlin was previously named Swift <img src="https://github.com/idaholab/marlin/blob/devel/doc/content/swift.png?raw=true" height="20pt">

## Contacts

The primary developer of Marlin is _Daniel Schwen_. Lattice Boltzmann implementation by _Nijat Rustamov_. In case of questions or problems please file a GitHub issue.

## Install

- MacOS and Linux [installation instructions](doc/content/installation.md)

## Why another spectral solver?

Marlin...

- ...uses familiar MOOSE input format
- ...can couple to MOOSE FE/FV models
- ...can utilize MOOSE postprocessing objects
- ...uses MOOSE input/outputs
- ...integrates with the MOOSE multiapp and transfers system (enabling it to work with stochastic tools to MCMC sample thousands of Marlin mesoscale models, perform inverse Bayesian inference, parameter calibration, UQ etc.)
- ...is fully device independent through libTorch
- ...has all its dependencies provided by MOOSE alone
- ...is focussed on ease of model development (with runtime parsed expressions, just-in-time compilation, and automatic dependency resolution)
- ...supports runing models on GPUs

### Other Software

Idaho National Laboratory is a cutting edge research facility which is a constantly producing high quality research and software. Feel free to take a look at our other software and scientific offerings at:

[Primary Technology Offerings Page](https://www.inl.gov/inl-initiatives/technology-deployment)

[Supported Open Source Software](https://github.com/idaholab)

[Raw Experiment Open Source Software](https://github.com/IdahoLabResearch)

[Unsupported Open Source Software](https://github.com/IdahoLabCuttingBoard)

### License

Copyright 2024 Battelle Energy Alliance, LLC

This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
