# LBMBGKCollision / LBMMRTCollision / LBMSmagorinskyCollision / LBMSmagorinskyMRTCollision

!syntax description /TensorComputes/Solve/LBMBGKCollision
!syntax description /TensorComputes/Solve/LBMMRTCollision
!syntax description /TensorComputes/Solve/LBMSmagorinskyCollision
!syntax description /TensorComputes/Solve/LBMSmagorinskyMRTCollision

This objects implements commonly used collision dynamics for Lattice Boltzmann simulation. Currently available collision operators are BGK single relaxation time and Multi Relaxation Time (MRT) operators. The additional operation is available to project the non-equilibrium distribution onto Hermite space to achieve stability with the boolean parameter `projection`. For high Reynolds number simulations, Smagorinsky LES collision dynamics is available for both BGK and MRT operators.

## Overview

The collision step relaxes the distribution functions towards their equilibrium state. This object provides a template for multiple LBM collision operators:

- **BGK**: Single-relaxation time model (`LBMBGKCollision`).
- **MRT**: Multi-relaxation time in moment space (`LBMMRTCollision`).
- **Smagorinsky**: Eddy-viscosity based turbulence models (`LBMSmagorinskyCollision`, `LBMSmagorinskyMRTCollision`).

Supply incoming and equilibrium distributions via
[!param](/TensorComputes/Solve/LBMBGKCollision/f) and
[!param](/TensorComputes/Solve/LBMBGKCollision/feq). Relaxation parameter related to viscous stress is controlled with
[!param](/TensorComputes/Solve/LBMBGKCollision/tau0); Smagorinsky constant can be controlled via
[!param](/TensorComputes/Solve/LBMSmagorinskyCollision/Cs).

### BGK Collision

The Bhatnagar-Gross-Krook (BGK) model uses a single relaxation time $\tau$ for all modes. The collision equation is:

$$
f_i^* = f_i - \frac{1}{\tau} (f_i - f_i^{eq})
$$

where $f_i^*$ is the post-collision distribution.

### MRT Collision

The Multi-Relaxation Time (MRT) model performs the collision in moment space, allowing different relaxation rates for different physical quantities (e.g., shear viscosity, bulk viscosity). The collision equation is:

$$
\mathbf{f}^* = \mathbf{f} - \mathbf{M}^{-1} \mathbf{S} \mathbf{M} (\mathbf{f} - \mathbf{f}^{eq})
$$

where:
- $\mathbf{M}$ is the transformation matrix from velocity space to moment space.
- $\mathbf{S}$ is the diagonal relaxation matrix containing relaxation rates for different moments.

### Smagorinsky LES Model

For high Reynolds number flows, the Smagorinsky Large Eddy Simulation (LES) model introduces an eddy viscosity $\nu_t$ to model subgrid-scale turbulence. This effectively modifies the relaxation time locally:

$$
\tau_{eff} = \tau_0 + \tau_{turb}
$$

The turbulent relaxation time $\tau_{turb}$ is computed based on the local strain rate tensor magnitude $|S|$ and the Smagorinsky constant $C_s$:

$$
\tau_{turb} = \frac{C_s \Delta x^2}{c_s^2} |S|
$$

The local strain rate magnitude $|S|$ is obtained by solving the following quadratic equation, which relates the non-equilibrium stress to the strain rate:

$$
t_{sgs}^2 |S|^2 + \tau_0 |S| = \frac{||\Pi^{neq}||}{\rho c_s^2}
$$

where $t_{sgs} = \frac{\sqrt{C_s} \Delta x}{c_s}$ and $||\Pi^{neq}||$ is the norm of the non-equilibrium second-order moment tensor.

This model is available for both BGK (`LBMSmagorinskyCollision`) and MRT (`LBMSmagorinskyMRTCollision`) operators. In the MRT case, only the relaxation rates corresponding to the stress tensor are modified.

### Hermite Regularization

To improve numerical stability, especially at high Reynolds numbers, the non-equilibrium part of the distribution function ($f_i^{neq} = f_i - f_i^{eq}$) can be projected onto the second-order Hermite polynomial space. This filters out higher-order "ghost" modes that do not contribute to the hydrodynamic limit but can cause instability.

The regularization procedure replaces $f_i^{neq}$ with its projection $f_i^{neq, reg}$:

$$
f_i^{neq, reg} = \frac{w_i}{2 c_s^2} \mathcal{H}_i^{(2)} : \mathbf{a}_1^{(2)}
$$

where the second-order Hermite polynomial tensor $\mathcal{H}_i^{(2)}$ and the corresponding non-equilibrium coefficient $\mathbf{a}_1^{(2)}$ are defined as:

$$
\mathcal{H}_i^{(2)} = \frac{\mathbf{e}_i \mathbf{e}_i}{c_s^2} - \mathbf{I}
$$

$$
\mathbf{a}_1^{(2)} = \sum_j f_j^{neq} \mathcal{H}_j^{(2)}
$$

This is enabled by setting `projection = true`.

## Example Input File Syntax

!listing
[TensorComputes/Solve]
  [collision_bgk]
    type = LBMBGKCollision
    buffer = fpc
    f = f
    feq = feq
    tau0 = 1.0
  []
[]
[TensorComputes/Solve]
  [collision_mrt]
    type = LBMSmagorinskyMRTCollision
    buffer = fpc
    f = f
    feq = feq
    tau0 = 0.5001
    Cs = 0.15
    projection=true
  []
[]
!listing-end

## LBMBGKCollision

!syntax parameters /TensorComputes/Solve/LBMBGKCollision

!syntax inputs /TensorComputes/Solve/LBMBGKCollision

!syntax children /TensorComputes/Solve/LBMBGKCollision

## LBMMRTCollision

!syntax parameters /TensorComputes/Solve/LBMMRTCollision

!syntax inputs /TensorComputes/Solve/LBMMRTCollision

!syntax children /TensorComputes/Solve/LBMMRTCollision

## LBMSmagorinskyCollision

!syntax parameters /TensorComputes/Solve/LBMSmagorinskyCollision

!syntax inputs /TensorComputes/Solve/LBMSmagorinskyCollision

!syntax children /TensorComputes/Solve/LBMSmagorinskyCollision

## LBMSmagorinskyMRTCollision

!syntax parameters /TensorComputes/Solve/LBMSmagorinskyMRTCollision

!syntax inputs /TensorComputes/Solve/LBMSmagorinskyMRTCollision

!syntax children /TensorComputes/Solve/LBMSmagorinskyMRTCollision
