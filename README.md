WaveFD.jl
=======
WaveFD.jl contains a reference collection of single time step wavefield propagation codes. 
The propagator kernels are written in c++, and we expose both c++ and Julia APIs for propagating wavefields. 

# Naming conventions
Propagator names use the following convention:
```
prop [2D / 3D] [Aco] [Iso / VTI / TTI] [Den] [Q] _ DE[O1 / O2] _ OPTIONAL FLAGS: [FD] [TD]
```
For examples consider the propagator names in the table below.

[insert mcells from Sam's benchmarks]

| Propagator | Dimension | Model Parameters | Isotropy | 
|:---|:--:|:---:|:---|
| Prop2DAcoIsoDenQ_DEO2_FDTD | 2D | v, ρ             | Isotropic |	
| Prop2DAcoVTIDenQ_DEO2_FDTD | 2D | v, ρ, ϵ, η       | Vertical transverse |	
| Prop2DAcoTTIDenQ_DEO2_FDTD | 2D | v, ρ, ϵ, η, θ, ϕ | Tilted transverse |
| Prop3DAcoIsoDenQ_DEO2_FDTD | 3D | v, ρ             | Isotropic  |
| Prop3DAcoVTIDenQ_DEO2_FDTD | 3D | v, ρ, ϵ, η       | Vertical transverse |
| Prop3DAcoTTIDenQ_DEO2_FDTD | 3D | v, ρ, ϵ, η, θ, ϕ | Tilted transverse |


# Note on unit tests and coverage
These single time step propagators are used to contruct nonlinear modeling operators and their
linearizations in the `JetPackWave` package. An extensive suite of unit tests guaranteeing
correctness is implemented in `JetPackWave`, and therefore is not re-implemented here. This
means that coverage statistics for tests in this package do not reflect the code as excersized
by the unit tests in `JetPackWave`.

# Notes
In addition to propagators, the Julia components of WaveFD.jl are meant to provide:
	* imaging conditions
	* wavelets
	* wavefield injection/extraction
	* wavefield compression (via CvxCompress)
	* absorbing boundary construction
	* extraction of a subset of an earth model for a given experiment aperture (ginsu)

