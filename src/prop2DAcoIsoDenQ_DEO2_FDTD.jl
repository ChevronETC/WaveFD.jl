mutable struct Prop2DAcoIsoDenQ_DEO2_FDTD
    p::Ptr{Cvoid}
end

function Prop2DAcoIsoDenQ_DEO2_FDTD(;
        nz=0,
        nx=0,
        nsponge=60,
        nbz=512,
        nbx=8,
        dz=5.0,
        dx=5.0,
        dt=0.001,
        nthreads=Sys.CPU_THREADS,
        freesurface=true,
        freqQ=5.0,
        qMin=0.1,
        qInterior=100.0)
    nz,nx,nsponge,nbz,nbx,nthreads = map(x->round(Int,x), (nz,nx,nsponge,nbz,nbx,nthreads))
    dz,dx,dt = map(x->Float32(x), (dz,dx,dt))

    @assert nx > 0
    @assert nz > 0
    @assert nsponge > 0
    @assert nthreads > 0
    @assert nbx > 0
    @assert nbz > 0

    fs = freesurface ? 1 : 0

    p = ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_alloc, libprop2DAcoIsoDenQ_DEO2_FDTD),
        Ptr{Cvoid},
        (Clong, Clong,    Clong, Clong, Clong,   Cfloat, Cfloat, Cfloat, Clong, Clong),
         fs,    nthreads, nx,    nz,    nsponge, dx,     dz,     dt,     nbx,   nbz)

    ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_SetupDtOmegaInvQ, libprop2DAcoIsoDenQ_DEO2_FDTD),
        Cvoid,
        (Ptr{Cvoid}, Cfloat, Cfloat, Cfloat),
         p,          freqQ,  qMin,   qInterior)

    Prop2DAcoIsoDenQ_DEO2_FDTD(p)
end

free(prop::Prop2DAcoIsoDenQ_DEO2_FDTD) = ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_free, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid, (Ptr{Cvoid},), prop.p)

function size(prop::Prop2DAcoIsoDenQ_DEO2_FDTD)
    nz = ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_getNz, libprop2DAcoIsoDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    nx = ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_getNx, libprop2DAcoIsoDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    (nz,nx)
end

for _f in (:V, :B, :PSpace, :PCur, :POld, :DtOmegaInvQ)
    symf = "Prop2DAcoIsoDenQ_DEO2_FDTD_get" * string(_f)
    @eval $(_f)(prop::Prop2DAcoIsoDenQ_DEO2_FDTD) = unsafe_wrap(Array, ccall(($symf, libprop2DAcoIsoDenQ_DEO2_FDTD), Ptr{Float32}, (Ptr{Cvoid},), prop.p), size(prop), own=false)
end

abstract type Prop2DAcoIsoDenQ_DEO2_FDTD_Model end
struct Prop2DAcoIsoDenQ_DEO2_FDTD_Model_V <: Prop2DAcoIsoDenQ_DEO2_FDTD_Model end
struct Prop2DAcoIsoDenQ_DEO2_FDTD_Model_B <: Prop2DAcoIsoDenQ_DEO2_FDTD_Model end
struct Prop2DAcoIsoDenQ_DEO2_FDTD_Model_VB <: Prop2DAcoIsoDenQ_DEO2_FDTD_Model end

propagateforward!(prop::Prop2DAcoIsoDenQ_DEO2_FDTD) = ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_TimeStep, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid, (Ptr{Cvoid},), prop.p)
propagateadjoint!(prop::Prop2DAcoIsoDenQ_DEO2_FDTD) = propagateforward!(prop) # self-adjoint

scale_spatial_derivatives!(prop::Prop2DAcoIsoDenQ_DEO2_FDTD) =
    ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_ScaleSpatialDerivatives, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid, (Ptr{Cvoid},), prop.p)

# v in model-space
function forwardBornInjection!(prop::Prop2DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop2DAcoIsoDenQ_DEO2_FDTD_Model_V, ic_weight::Real, dmodel, wavefields)
    ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_ForwardBornInjection_V, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},          Cfloat),
         prop.p,     dmodel["v"], wavefields["pspace"], ic_weight)
end

# v,b in model-space
function forwardBornInjection!(prop::Prop2DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop2DAcoIsoDenQ_DEO2_FDTD_Model_VB, dmodel, wavefields)
    ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_ForwardBornInjection_VB, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},     Ptr{Cfloat}),
         prop.p,     dmodel["v"], dmodel["b"], wavefields["pold"], wavefields["pspace"])
end

# b in model-space
function forwardBornInjection!(prop::Prop2DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop2DAcoIsoDenQ_DEO2_FDTD_Model_B, dmodel, wavefields)
    ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_ForwardBornInjection_B, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},     Ptr{Cfloat}),
         prop.p,     dmodel["b"], wavefields["pold"], wavefields["pspace"])
end

# v in model-space
function adjointBornAccumulation!(prop::Prop2DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop2DAcoIsoDenQ_DEO2_FDTD_Model_V, imagingcondition::ImagingConditionStandard, dmodel, wavefields)
    ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_V, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},        ),
         prop.p,     dmodel["v"], wavefields["pspace"])
end

# v,b in model-space
function adjointBornAccumulation!(prop::Prop2DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop2DAcoIsoDenQ_DEO2_FDTD_Model_VB, imagingcondition::ImagingConditionStandard, dmodel, wavefields)
    ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_VB, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},     Ptr{Cfloat}),
         prop.p,     dmodel["v"], dmodel["b"], wavefields["pold"], wavefields["pspace"])
end

# b in model-space
function adjointBornAccumulation!(prop::Prop2DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop2DAcoIsoDenQ_DEO2_FDTD_Model_B, imagingcondition::ImagingConditionStandard, dmodel, wavefields)
    ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_B, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},     Ptr{Cfloat}),
         prop.p,     dmodel["b"], wavefields["pold"], wavefields["pspace"])
end

# v in model-space with FWI wavefield separation
function adjointBornAccumulation!(prop::Prop2DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop2DAcoIsoDenQ_DEO2_FDTD_Model_V, imagingcondition::ImagingConditionWaveFieldSeparationFWI, ic_weight::Real, dmodel, wavefields)
    ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_wavefieldsep, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},      Clong, Cfloat),
         prop.p,     dmodel["v"], wavefields["pspace"], 1, ic_weight)
 end

# v in model-space with RTM wavefield separation
function adjointBornAccumulation!(prop::Prop2DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop2DAcoIsoDenQ_DEO2_FDTD_Model_V, imagingcondition::ImagingConditionWaveFieldSeparationRTM, ic_weight::Real, dmodel, wavefields)
    ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_wavefieldsep, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},      Clong, Cfloat),
         prop.p,     dmodel["v"], wavefields["pspace"], 0, ic_weight)
 end

# v in model-space with MIX wavefield separation 
# function adjointBornAccumulation!(prop::Prop2DAcoIsoDenQ_DEO2_FDTD, modeltype::Prop2DAcoIsoDenQ_DEO2_FDTD_Model_V, imagingcondition::ImagingConditionWaveFieldSeparationMIX, RTM_weight::Real, dmodel, wavefields)

#     dmodel_FWI = similar(dmodel["v"])
#     dmodel_FWI .= dmodel["v"].

#     dmodel_RTM = similar(dmodel["v"])
#     dmodel_RTM .= dmodel["v"]

#     # FWI IC 
#     ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_wavefieldsep, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid,
#      (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},          Clong, Cfloat),
#       prop.p,     dmodel_FWI, wavefields["pspace"], 1,     1.0f0)
#     # RTM IC 
#     ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_AdjointBornAccumulation_wavefieldsep, libprop2DAcoIsoDenQ_DEO2_FDTD), Cvoid,
#      (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},      Clong, Cfloat),
#       prop.p,     dmodel_RTM, wavefields["pspace"], 0, 1.0f0)

#     dmodel["v"] .+= (1-RTM_weight) .* (dmodel_FWI ./ maximum(abs,dmodel_FWI)) .+ RTM_weight .* (dmodel_RTM ./ maximum(abs,dmodel_RTM))

#  end

function show(io::IO, prop::Prop2DAcoIsoDenQ_DEO2_FDTD)
    nx = ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_getNx, libprop2DAcoIsoDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    nz = ccall((:Prop2DAcoIsoDenQ_DEO2_FDTD_getNz, libprop2DAcoIsoDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    write(io, "Prop2DAcoIsoDenQ_DEO2_FDTD -- nx,nz; $nx,$nz")
end
