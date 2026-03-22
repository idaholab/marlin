# LBMSpecularReflectionBoundary

!syntax description /TensorComputes/Boundary/LBMSpecularReflectionBoundary

This compute object implements a combination of bounce-back and specular reflection boundary conditions for rarefied gas flows in Lattice Boltzmann simulations. It is designed for use with complex solid geometries defined through `binary_media` and is currently limited to D2Q9 stencils.

## Overview

At each boundary node (fluid cell adjacent to a solid), the incoming distribution is split into two parts:

$$ f_{\mathrm{opposite}} \mathrel{+}= r \cdot f_{\mathrm{old,in}} \quad \text{(bounce-back)} $$

$$ f_{\mathrm{specular}} \mathrel{+}= (1 - r) \cdot f_{\mathrm{old,in}} \quad \text{(specular reflection)} $$

where $r$ is a per-node combination coefficient computed from the local Knudsen number:

$$ \sigma = 1 - \log_{10}(1 + \text{Kn}^{0.7})$$
$$ \sigma_v = \frac{2 - \sigma}{\sigma}$$
$$ A_1 = 1 - 0.1817 \, \sigma_v $$
$$ r = \frac{1}{1 + \sqrt{\pi / 6} \, A_1 \, \sigma_v} $$

When $r = 1$ the scheme reduces to pure bounce-back; when $r = 0$ it is pure specular reflection.

### Boundary Type Classification

Each boundary node is classified by encoding the connectivity of its 9 D2Q9 neighbors (fluid vs. solid) into a 9-bit binary number. A precomputed lookup table of 52 known boundary types maps this code to the specular reflection direction for each lattice velocity. Directions where streaming is not allowed (solid neighbor) receive the combined bounce-back / specular treatment.

### Requirements

- A `binary_media` buffer must be defined in `LatticeBoltzmannProblem` to identify solid and fluid regions.
- Only `boundary = wall` is supported (domain-edge boundaries such as `left`, `right`, etc. are not handled by this object).
- Only D2Q9 stencils are currently supported.

## Example Input File Syntax

!listing examples/lbm/rarefied_gas/channel.i block=TensorComputes/Boundary/wall

!syntax parameters /TensorComputes/Boundary/LBMSpecularReflectionBoundary

!syntax inputs /TensorComputes/Boundary/LBMSpecularReflectionBoundary

!syntax children /TensorComputes/Boundary/LBMSpecularReflectionBoundary
