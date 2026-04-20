# FFTGradient

!syntax description /TensorComputes/Solve/FFTGradient

Compute the gradient component of a real space tensor along a selected axis using the
spectral gradient. Internally this involves a forward and a backward Fourier transform
and inbetween a multiplication with the $\vec k$ vector and the imaginary unit $i$.

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [to_k]
      type = ForwardFFT
      buffer = cbar
      input = c
    []
    [dc_dx]
      type = FFTGradient
      buffer = dc_dx
      input = c
      direction = 0                    # 0:x, 1:y, 2:z
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/FFTGradient

!syntax inputs /TensorComputes/Solve/FFTGradient

!syntax children /TensorComputes/Solve/FFTGradient
