module WaveFD

using Base.Threads, CvxCompress, DSP, Distributed, DistributedArrays, FFTW, LinearAlgebra, NearestNeighbors, Random, SpecialFunctions, StaticArrays, Statistics, WaveFD_jll

import
Base.convert,
Base.copy,
Base.get,
Base.min,
Base.max,
Base.maximum,
Base.show,
Base.size

abstract type Language end
struct LangC <: Language end
struct LangJulia <: Language end
show(io::IO, l::LangC) = write(io, "C")
show(io::IO, l::LangJulia) = write(io, "Julia")

abstract type ImagingCondition end
struct ImagingConditionStandard <: ImagingCondition end
struct ImagingConditionWaveFieldSeparationFWI <: ImagingCondition end
struct ImagingConditionWaveFieldSeparationRTM <: ImagingCondition end
struct ImagingConditionWaveFieldSeparationMIX <: ImagingCondition end

include("stencil.jl")
include("spacetime.jl")
include("compressedio.jl")
include("wavelet.jl")
include("absorb.jl")
include("illumination.jl")

include("prop2DAcoIsoDenQ_DEO2_FDTD.jl")
include("prop2DAcoTTIDenQ_DEO2_FDTD.jl")
include("prop2DAcoVTIDenQ_DEO2_FDTD.jl")

include("prop3DAcoIsoDenQ_DEO2_FDTD.jl")
include("prop3DAcoTTIDenQ_DEO2_FDTD.jl")
include("prop3DAcoVTIDenQ_DEO2_FDTD.jl")

export
dtmod,
Prop2DAcoIsoDenQ_DEO2_FDTD,
Prop2DAcoTTIDenQ_DEO2_FDTD,
Prop2DAcoVTIDenQ_DEO2_FDTD,
Prop3DAcoIsoDenQ_DEO2_FDTD,
Prop3DAcoTTIDenQ_DEO2_FDTD,
Prop3DAcoVTIDenQ_DEO2_FDTD,
fieldfile!,
free,
Ginsu,
linearadjoint,
linearforward,
nonlinearforward,
ntmod,
reportfreq!,
sourceillum!,
sourceillum,
traces,
Wavelet,
WaveletDerivRicker,
WaveletMinPhaseRicker,
WaveletCausalRicker,
WaveletOrmsby,
WaveletMinPhaseOrmsby,
WaveletCausalOrmsby,
WaveletRicker,
WaveletSine

end
