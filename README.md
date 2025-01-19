# SI-SR

## Overview
Discovering governing equations that describe the behaviour of dynamic systems is a crucial and challenging task in
the fields of physics and engineering. This work introduces a novel symmetry-inspired symbolic regression (SI-SR)
method for discovering the governing differential equations of spatio-temporal systems from noisy and sparse data. SI-
SR uses deep neural networks for robust derivative estimation, symbolic regression for rich non-linear representations,
and a symmetry method for recursively adjusting the candidate function library by identifying symmetry invariants in
the underlying equation, thereby enabling the discovery of accurate and concise models. The method constructs sym-
metric function libraries based on identified properties, distinguishing the intrinsic structure of the governing equations
from redundant terms, while also adjusting the model coefficients. We demonstrate the effectiveness and robustness
of our approach by identifying canonical partial differential equation systems with constant or variable coefficients
under varying noise levels across multiple scientific fields. The resulting framework shows the capability of symmetry
information in discovering concise and accurate models when data is noisy and scarce.
This Repository provide the code and data for the following article
* Symmetry-inspired learning of governing equations from noisy data

## Requirements
### Hardware requirements
All tests were performed on a computer with 
* an AMD Ryzen 7 7800X3D 8-Core CPU, 32.0 GB RAM, and
* an NVIDIA GeForce RTX 4070 Ti GPU.
### Software requirements
We strongly recommend using Anaconda for environment management and running examples with Jupyter. The packages used in this work are as follows:
* NVIDIA® GPU drivers
* CUDA® Toolkit*
* Anaconda
* Jupyter
* Python 3.11
* PyTorch 2.22
* NumPy
* SciPy
* Matplotlib

## Run example
To run our examples, you can visit each corresponding files and run notebook in jupyter. The examples include:
* Burgers equation (Burgers)
* Korteweg–de Vries equation (KdV)
* FitzHugh–Nagumo type reaction–diffusion equation (FN)
* Navier-Stokes vorticity equation (NS)
* Spatially dependent advection-diffusion equation (AD)
* Temporally dependent Burgers equation (Burgers_variable)

Datasets are included in the files, except for the FN, NS, and RD examples. For these cases, we provide Dropbox links for access, as the datasets exceed 100 MB.
