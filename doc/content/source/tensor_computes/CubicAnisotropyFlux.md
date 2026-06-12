# CubicAnisotropyFlux

!syntax description /TensorComputes/*/CubicAnisotropyFlux

## Overview

This compute calculates the full anisotropic gradient energy flux for cubic crystal symmetry, including the critical "corner correction" term that is essential for dendritic morphology formation.

## Theory

For an anisotropic phase-field model, the gradient energy is:

\begin{equation}
E_{\text{grad}} = \frac{W^2}{2} \int a^2(\mathbf{n}) |\nabla\phi|^2 \, dV
\end{equation}

where $a(\mathbf{n})$ is the anisotropy function depending on the interface normal $\mathbf{n} = \nabla\phi / |\nabla\phi|$.

The variational derivative gives:

\begin{equation}
\frac{\delta E}{\delta \phi} = -W^2 \nabla \cdot \left( a^2 \nabla\phi + \frac{1}{2}|\nabla\phi|^2 \frac{\partial a^2}{\partial \nabla\phi} \right)
\end{equation}

The second term in the divergence, often called the "corner correction" or "anisotropy flux correction," is frequently omitted in simplified implementations but is essential for:

1. Proper dendrite tip formation and growth
2. Correct crystallographic orientation selection
3. Numerical stability with strong anisotropy

## Cubic Anisotropy Function

For cubic symmetry, the anisotropy function is:

\begin{equation}
a = 1 + \epsilon_a \left( \frac{q_x^4 + q_y^4 + q_z^4}{|\mathbf{q}|^4} - 0.6 \right)
\end{equation}

where $\mathbf{q} = R^T \nabla\phi$ is the gradient rotated into the crystal reference frame, and $R$ is the rotation matrix defining the grain orientation.

The offset of 0.6 ensures that $a = 1$ on average over all orientations in 3D.

## Output

This compute outputs the flux vector:

\begin{equation}
\mathbf{F} = (a^2 - 1)\nabla\phi + \frac{1}{2}|\nabla\phi|^2 \frac{\partial a^2}{\partial \nabla\phi}
\end{equation}

The $(a^2-1)$ form is used because the isotropic part ($\nabla\phi$) is handled by the linear implicit solver.

## Example Input File Syntax

!listing examples/thermal_multigrain_dendrites.i block=TensorComputes/Solve/fluxvec1

!syntax parameters /TensorComputes/*/CubicAnisotropyFlux

!syntax inputs /TensorComputes/*/CubicAnisotropyFlux
