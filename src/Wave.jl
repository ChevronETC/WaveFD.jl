module Wave

const _jl_libprop2DAcoIsoDenQ_DEO2_FDTD = normpath(joinpath(Base.source_path(),"../../deps/usr/lib/libprop2DAcoIsoDenQ_DEO2_FDTD"))
const _jl_libprop2DAcoTTIDenQ_DEO2_FDTD = normpath(joinpath(Base.source_path(),"../../deps/usr/lib/libprop2DAcoTTIDenQ_DEO2_FDTD"))
const _jl_libprop2DAcoVTIDenQ_DEO2_FDTD = normpath(joinpath(Base.source_path(),"../../deps/usr/lib/libprop2DAcoVTIDenQ_DEO2_FDTD"))

const _jl_libprop3DAcoIsoDenQ_DEO2_FDTD = normpath(joinpath(Base.source_path(),"../../deps/usr/lib/libprop3DAcoIsoDenQ_DEO2_FDTD"))
const _jl_libprop3DAcoTTIDenQ_DEO2_FDTD = normpath(joinpath(Base.source_path(),"../../deps/usr/lib/libprop3DAcoTTIDenQ_DEO2_FDTD"))
const _jl_libprop3DAcoVTIDenQ_DEO2_FDTD = normpath(joinpath(Base.source_path(),"../../deps/usr/lib/libprop3DAcoVTIDenQ_DEO2_FDTD"))

const _jl_libillumination               = normpath(joinpath(Base.source_path(),"../../deps/usr/lib/libillumination"))
const _jl_libspacetime                  = normpath(joinpath(Base.source_path(),"../../deps/usr/lib/libspacetime"))

using Base.Threads, CvxCompress, DSP, Distributed, DistributedArrays, FFTW, LinearAlgebra, NearestNeighbors, SpecialFunctions, StaticArrays, Statistics

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
WaveletRicker,
WaveletSine

end